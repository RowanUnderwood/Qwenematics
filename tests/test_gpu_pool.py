import asyncio

import pytest

from harness.jobs.gpu_pool import KREA, MINIMAX, GpuPool


def make(backlog=1, borrow=True, loaded_video=None, min_backlog=2, queued=0):
    frees = []
    st = {"backlog": backlog, "borrow": borrow}

    async def free(name):
        frees.append(name)

    pool = GpuPool(free=free, can_borrow=lambda: st["borrow"], backlog=lambda: st["backlog"],
                   min_backlog=lambda: min_backlog, queued=lambda: queued, est={"image": 33.0, "video": 11.0})
    pool.loaded["video"] = loaded_video
    return pool, frees, st


async def take(pool, out, hold: asyncio.Event):
    async with pool.image_slot() as server:
        out.append(server)
        await hold.wait()


def run(coro):
    return asyncio.run(coro)


def test_single_image_stays_on_image_gpu_unless_krea_is_warm_there():
    async def go():
        pool, frees, _ = make(backlog=1)
        async with pool.image_slot() as s:
            assert s == "image"
        pool, frees, _ = make(backlog=1, loaded_video=KREA)
        async with pool.image_slot() as s:
            assert s == "video"
        assert frees == []                       # already Krea: nothing to empty
    run(go())


def test_single_image_borrows_when_threshold_is_one():
    async def go():
        pool, frees, _ = make(backlog=1, min_backlog=1)
        async with pool.image_slot() as s:
            assert s == "video"
        assert frees == ["video"] and pool.loaded["video"] == KREA
    run(go())


def test_cold_load_does_not_skew_the_estimate():
    async def go():
        pool, _, _ = make(backlog=1, min_backlog=1)
        async with pool.image_slot() as s:
            await asyncio.sleep(0.02)
        assert s == "video" and pool.est["video"] == 11.0       # cold: not averaged in
        async with pool.image_slot():
            await asyncio.sleep(0.02)
        assert pool.est["video"] < 11.0                         # warm: averaged in
    run(go())


def test_small_batch_stays_on_fast_gpu():
    async def go():
        pool, _, _ = make(backlog=3, min_backlog=1)
        hold = asyncio.Event()
        out: list[str] = []
        tasks = [asyncio.create_task(take(pool, out, hold)) for _ in range(3)]
        await asyncio.sleep(0.05)
        assert out == ["video"]                  # 11 s left + 2 x 11 s isn't more than 33 s
        hold.set()
        await asyncio.gather(*tasks)
        assert out == ["video"] * 3
    run(go())


def test_long_batch_uses_both_gpus_even_with_few_jobs_in_flight():
    async def go():
        pool, _, _ = make(backlog=8, min_backlog=1, queued=6)
        hold = asyncio.Event()
        out: list[str] = []
        tasks = [asyncio.create_task(take(pool, out, hold)) for _ in range(2)]
        await asyncio.sleep(0.05)
        assert sorted(out) == ["image", "video"]
        hold.set()
        await asyncio.gather(*tasks)
    run(go())


def test_jobs_past_their_render_dont_send_the_last_image_to_the_slow_gpu():
    async def go():
        pool, _, _ = make(backlog=6, min_backlog=1)     # 4 of them are in the critic
        hold = asyncio.Event()
        out: list[str] = []
        tasks = [asyncio.create_task(take(pool, out, hold)) for _ in range(2)]
        await asyncio.sleep(0.05)
        assert out == ["video"]
        hold.set()
        await asyncio.gather(*tasks)
        assert out == ["video", "video"]
    run(go())


def test_borrow_off_or_unavailable():
    async def go():
        pool, _, _ = make(backlog=5, borrow=False, loaded_video=KREA)
        async with pool.image_slot() as s:
            assert s == "image"
    run(go())


def test_batch_prefers_fast_gpu_and_overflows_to_slow_one():
    async def go():
        pool, frees, _ = make(backlog=4)
        hold = asyncio.Event()
        out: list[str] = []
        tasks = [asyncio.create_task(take(pool, out, hold)) for _ in range(4)]
        await asyncio.sleep(0.05)
        # first goes to the 5090 (emptied of MiniMax once); waiters 2 and 3 are better off
        # queueing for it (11/22 s + remaining < 33 s), the 4th takes the 3090
        assert sorted(out) == ["image", "video"]
        assert frees == ["video"] and pool.loaded["video"] == KREA
        hold.set()
        await asyncio.gather(*tasks)
        assert out.count("video") == 3 and out.count("image") == 1
    run(go())


def test_video_first_waits_for_running_image_then_empties_krea():
    async def go():
        pool, frees, _ = make(backlog=3, loaded_video=KREA)
        hold = asyncio.Event()
        out: list[str] = []
        t1 = asyncio.create_task(take(pool, out, hold))
        await asyncio.sleep(0.01)
        assert out == ["video"]
        got_video = asyncio.Event()

        async def shot():
            async with pool.video_slot():
                got_video.set()
                # while the video holds the 5090, new images go to the 3090
                async with pool.image_slot() as s:
                    assert s == "image"
        t2 = asyncio.create_task(shot())
        await asyncio.sleep(0.02)
        assert not got_video.is_set()            # waits for the image already on the 5090
        assert not pool.borrowable()             # but no new borrow starts meanwhile
        hold.set()
        await asyncio.gather(t1, t2)
        assert got_video.is_set()
        assert frees == ["video"] and pool.loaded["video"] == MINIMAX
    run(go())


def test_cancelled_waiter_is_dropped():
    async def go():
        pool, _, _ = make(backlog=1)
        hold = asyncio.Event()
        out: list[str] = []
        t1 = asyncio.create_task(take(pool, out, hold))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(take(pool, out, hold))
        await asyncio.sleep(0.01)
        t2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t2
        assert pool.status()["waiting_images"] == 0
        hold.set()
        await t1
        assert pool.busy == {"image": False, "video": False}
    run(go())
