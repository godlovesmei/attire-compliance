"""Smoothing temporal berbasis voting mayoritas per `track_id` untuk mencegah flicker UI."""

from __future__ import annotations

from collections import Counter, deque

from attire_compliance.pipeline.rules import ComplianceStatus


class TemporalSmoother:
    """Haluskan keputusan kepatuhan per track selama jendela `window_size` frame terakhir."""

    def __init__(self, window_size: int = 15, max_stale_frames: int = 90) -> None:
        if window_size < 1:
            raise ValueError("window_size harus >= 1")
        self.window_size = window_size
        self.max_stale_frames = max_stale_frames
        self._frame_counter = 0
        self._history: dict[int, deque[ComplianceStatus]] = {}
        self._last_seen: dict[int, int] = {}

    def step_frame(self) -> None:
        """Majukan counter frame & bersihkan track yang sudah lama hilang."""
        self._frame_counter += 1
        stale = [
            tid
            for tid, seen_at in self._last_seen.items()
            if (self._frame_counter - seen_at) > self.max_stale_frames
        ]
        for tid in stale:
            self._history.pop(tid, None)
            self._last_seen.pop(tid, None)

    def update(self, track_id: int, status: ComplianceStatus) -> ComplianceStatus:
        """Tambahkan status terbaru untuk `track_id` dan kembalikan status mayoritas."""
        hist = self._history.setdefault(track_id, deque(maxlen=self.window_size))
        hist.append(status)
        self._last_seen[track_id] = self._frame_counter

        counts = Counter(hist)
        max_votes = max(counts.values())
        tied = {st for st, cnt in counts.items() if cnt == max_votes}
        if status in tied:
            return status
        # Jika seri tanpa melibatkan status terbaru, ambil yang paling akhir muncul
        for prev in reversed(hist):
            if prev in tied:
                return prev
        return status

    def reset(self) -> None:
        """Kosongkan seluruh riwayat track (dipanggil saat tombol Reset ditekan)."""
        self._frame_counter = 0
        self._history.clear()
        self._last_seen.clear()
