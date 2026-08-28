# 依赖安装（在项目根目录终端执行）：
# pip install basic-pitch pretty_midi
# 若无法读取 .m4a，请先安装 FFmpeg 并确保已加入系统 PATH。

"""哼唱转 MIDI：按乐理谐和度推荐和弦走向，并合并导出钢琴伴奏。"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pretty_midi
from basic_pitch.inference import predict

AUDIO_FILENAME = "humming.wav.m4a"

DEFAULT_TEMPO_BPM = 120.0
BEATS_PER_CHORD = 4
CHORD_VELOCITY = 64
PIANO_PROGRAM = 0  # Acoustic Grand Piano

# 谐和度权重：根音 / 三音 / 五音 / 七音
ROLE_WEIGHTS = {"root": 1.00, "3rd": 0.92, "7th": 0.84, "5th": 0.72}
LONG_NOTE_SECONDS = 0.45
STRONG_BEAT_WINDOW_RATIO = 0.18  # 相对一拍时长，落在强拍附近
CLASH_SEMITONE = 1
TOP_N = 4

_N = pretty_midi.note_name_to_number


def _voicing(*names: str) -> list[int]:
    return [_N(name) for name in names]


@dataclass(frozen=True)
class ChordProgression:
    id: str
    name: str
    emotion: str
    chords_midi: list[list[int]]
    vibe_desc: str


# 10～15 种经典走向；chords_midi 为钢琴音区 MIDI 音符（含三度/七度以便评分）
CHORD_LIBRARY: list[ChordProgression] = [
    ChordProgression(
        id="pop_1564",
        name="流行王道 1-5-6-4",
        emotion="阳光治愈",
        chords_midi=[
            _voicing("C3", "E3", "G3", "C4"),
            _voicing("G2", "B2", "D3", "G3"),
            _voicing("A2", "C3", "E3", "A3"),
            _voicing("F2", "A2", "C3", "F3"),
        ],
        vibe_desc="明亮大调循环，适合清爽主歌与副歌，像午后流行金曲。",
    ),
    ChordProgression(
        id="sad_4536",
        name="伤感 4-5-3-6",
        emotion="深夜伤感",
        chords_midi=[
            _voicing("F2", "A2", "C3", "F3"),
            _voicing("G2", "B2", "D3", "G3"),
            _voicing("E2", "G2", "B2", "E3"),
            _voicing("A2", "C3", "E3", "A3"),
        ],
        vibe_desc="先扬后抑的 IV–V–iii–vi，适合夜色抒情与未说出口的情绪。",
    ),
    ChordProgression(
        id="jazz_251",
        name="爵士 2-5-1",
        emotion="松弛爵士",
        chords_midi=[
            _voicing("D2", "F2", "A2", "C3"),
            _voicing("G2", "B2", "D3", "F3"),
            _voicing("C2", "E2", "G2", "B2"),
            _voicing("C2", "E2", "G2", "B2"),
        ],
        vibe_desc="经典 ii–V–I 七和弦解决，松弛、有呼吸感，适合即兴哼唱。",
    ),
    ChordProgression(
        id="urban_rnb",
        name="都市 R&B 6-4-1-5",
        emotion="都市夜色",
        chords_midi=[
            _voicing("A2", "C3", "E3", "G3"),
            _voicing("F2", "A2", "C3", "E3"),
            _voicing("C3", "E3", "G3", "B3"),
            _voicing("G2", "B2", "D3", "F3"),
        ],
        vibe_desc="七和弦铺底的城市节奏，适合夜驾、慢板 R&B 与口语化旋律。",
    ),
    ChordProgression(
        id="dark_epic",
        name="暗黑史诗 6-5-4-3",
        emotion="暗黑史诗",
        chords_midi=[
            _voicing("A2", "C3", "E3", "A3"),
            _voicing("G2", "B2", "D3", "G3"),
            _voicing("F2", "A2", "C3", "F3"),
            _voicing("E2", "G2", "B2", "E3"),
        ],
        vibe_desc="下行小调进行，张力持续堆积，适合史诗叙事与阴郁高潮。",
    ),
    ChordProgression(
        id="dream_city",
        name="梦幻都市 1Δ7-3m7-6m7-5",
        emotion="梦幻霓虹",
        chords_midi=[
            _voicing("C3", "E3", "G3", "B3"),
            _voicing("E2", "G2", "B2", "D3"),
            _voicing("A2", "C3", "E3", "G3"),
            _voicing("G2", "B2", "D3", "G3"),
        ],
        vibe_desc="大七与小七交替，空气感强，适合霓虹、梦核与都市民谣。",
    ),
    ChordProgression(
        id="canon",
        name="卡农进行 1-5-6-3-4-1-4-5",
        emotion="温馨怀旧",
        chords_midi=[
            _voicing("C3", "E3", "G3", "C4"),
            _voicing("G2", "B2", "D3", "G3"),
            _voicing("A2", "C3", "E3", "A3"),
            _voicing("E2", "G2", "B2", "E3"),
            _voicing("F2", "A2", "C3", "F3"),
            _voicing("C3", "E3", "G3", "C4"),
            _voicing("F2", "A2", "C3", "F3"),
            _voicing("G2", "B2", "D3", "G3"),
        ],
        vibe_desc="帕赫贝尔式循环，庄重又亲切，适合婚礼、回忆杀与长线条哼唱。",
    ),
    ChordProgression(
        id="youth_6415",
        name="青春热血 6-4-1-5",
        emotion="青春热血",
        chords_midi=[
            _voicing("A2", "C3", "E3", "A3"),
            _voicing("F2", "A2", "C3", "F3"),
            _voicing("C3", "E3", "G3", "C4"),
            _voicing("G2", "B2", "D3", "G3"),
        ],
        vibe_desc="华语流行小调王道，从压抑推向释放，适合青春叙事副歌。",
    ),
    ChordProgression(
        id="tragic_6413",
        name="悲情叙事 6-4-1-3",
        emotion="悲情叙事",
        chords_midi=[
            _voicing("A2", "C3", "E3", "A3"),
            _voicing("F2", "A2", "C3", "F3"),
            _voicing("C3", "E3", "G3", "C4"),
            _voicing("E2", "G#2", "B2", "E3"),
        ],
        vibe_desc="结尾大三和弦（V/vi）制造戏剧转折，适合悲剧告白与电影感桥段。",
    ),
    ChordProgression(
        id="highway_mixo",
        name="公路自由 1-b7-4-5",
        emotion="公路自由",
        chords_midi=[
            _voicing("C3", "E3", "G3", "C4"),
            _voicing("Bb2", "D3", "F3", "Bb3"),
            _voicing("F2", "A2", "C3", "F3"),
            _voicing("G2", "B2", "D3", "G3"),
        ],
        vibe_desc="混合利底亚/摇滚色彩，开阔、带风声，适合公路与复古摇滚哼唱。",
    ),
    ChordProgression(
        id="neo_soul",
        name="新灵魂 2m9-5-1Δ9-6m7",
        emotion="细腻灵魂",
        chords_midi=[
            _voicing("D2", "F2", "A2", "C3", "E3"),
            _voicing("G2", "B2", "D3", "F3", "E3"),
            _voicing("C2", "E2", "G2", "B2", "D3"),
            _voicing("A2", "C3", "E3", "G3"),
        ],
        vibe_desc="九和弦色彩的 Neo-Soul，细腻、偏夜店Lounge，适合气息绵长的旋律。",
    ),
    ChordProgression(
        id="cinematic_minor",
        name="电影小调 1-b6-b3-b7",
        emotion="史诗苍茫",
        chords_midi=[
            _voicing("C2", "Eb2", "G2", "C3"),
            _voicing("Ab2", "C3", "Eb3", "Ab3"),
            _voicing("Eb2", "G2", "Bb2", "Eb3"),
            _voicing("Bb2", "D3", "F3", "Bb3"),
        ],
        vibe_desc="自然小调史诗进行，辽阔、空镜感强，适合预告片式哼唱。",
    ),
    ChordProgression(
        id="bossa_2516",
        name="慵懒波萨 2-5-1-6",
        emotion="慵懒波萨",
        chords_midi=[
            _voicing("D2", "F2", "A2", "C3"),
            _voicing("G2", "B2", "D3", "F3"),
            _voicing("C2", "E2", "G2", "B2"),
            _voicing("A2", "C#3", "E3", "G3"),
        ],
        vibe_desc="Bossa 常用 ii–V–I–VI7，摇摆而慵懒，适合咖啡馆与夏夜窗边。",
    ),
    ChordProgression(
        id="moonlight_em",
        name="清冷月光 3-1-2-7",
        emotion="清冷月光",
        chords_midi=[
            _voicing("E2", "G2", "B2", "E3"),
            _voicing("C3", "E3", "G3", "C4"),
            _voicing("D3", "F#3", "A3", "D4"),
            _voicing("B2", "D3", "F#3", "B3"),
        ],
        vibe_desc="E 小调民谣进行，清冷克制，适合月下独白与轻声哼鸣。",
    ),
    ChordProgression(
        id="romance_1645",
        name="浪漫告白 1-6-2-5",
        emotion="浪漫告白",
        chords_midi=[
            _voicing("C3", "E3", "G3", "C4"),
            _voicing("A2", "C3", "E3", "A3"),
            _voicing("D2", "F2", "A2", "D3"),
            _voicing("G2", "B2", "D3", "G3"),
        ],
        vibe_desc="I–vi–ii–V 圆舞告白感（4/4 铺底），温柔推进，适合情歌主歌。",
    ),
]


@dataclass(frozen=True)
class MelodyNote:
    pitch: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class ScoredProgression:
    progression: ChordProgression
    score: float


def estimate_tempo_bpm(midi: pretty_midi.PrettyMIDI) -> float:
    try:
        _times, tempos = midi.get_tempo_changes()
        if len(tempos) > 0 and float(tempos[0]) > 0:
            return float(tempos[0])
    except Exception:
        pass
    try:
        estimated = midi.estimate_tempo()
        if estimated and estimated > 0:
            return float(estimated)
    except Exception:
        pass
    return DEFAULT_TEMPO_BPM


def extract_melody_notes(midi: pretty_midi.PrettyMIDI) -> list[MelodyNote]:
    notes: list[MelodyNote] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.end > note.start:
                notes.append(MelodyNote(int(note.pitch), float(note.start), float(note.end)))
    notes.sort(key=lambda n: (n.start, n.end, n.pitch))
    return notes


def melody_end_time(notes: Sequence[MelodyNote], midi: pretty_midi.PrettyMIDI) -> float:
    if notes:
        return max(n.end for n in notes)
    return float(midi.get_end_time() or 0.0)


def chord_tone_roles(chords_midi: Sequence[int]) -> dict[int, str]:
    """根据最低音为根，识别 Root / 3rd / 5th / 7th 的音级（pitch class 0-11）。"""
    if not chords_midi:
        return {}
    root = min(chords_midi) % 12
    pcs = {p % 12 for p in chords_midi}
    roles: dict[int, str] = {root: "root"}

    third = next((pc for pc in ((root + 4) % 12, (root + 3) % 12) if pc in pcs and pc != root), None)
    if third is not None:
        roles[third] = "3rd"

    fifth = next(
        (pc for pc in ((root + 7) % 12, (root + 6) % 12, (root + 8) % 12) if pc in pcs and pc not in roles),
        None,
    )
    if fifth is not None:
        roles[fifth] = "5th"

    seventh = next(
        (pc for pc in ((root + 10) % 12, (root + 11) % 12) if pc in pcs and pc not in roles),
        None,
    )
    if seventh is not None:
        roles[seventh] = "7th"

    return roles


def _pc_distance(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def _is_strong_beat(onset: float, window_start: float, beat_len: float) -> bool:
    """小节内第 1 拍（和弦窗起点）附近的起音视为强拍。"""
    if beat_len <= 0:
        return False
    delta = abs(onset - window_start)
    return delta <= beat_len * STRONG_BEAT_WINDOW_RATIO


def _segment_harmony_points(
    pitch: int,
    overlap: float,
    is_long: bool,
    is_strong: bool,
    roles: dict[int, str],
) -> tuple[float, float]:
    """返回 (加权得分贡献, 权重质量)。"""
    mass = overlap
    if is_long:
        mass *= 1.55
    if is_strong:
        mass *= 1.45

    pc = pitch % 12
    role = roles.get(pc)
    chord_pcs = set(roles.keys())

    if role in ROLE_WEIGHTS:
        # 长音/强拍落在和弦内音再加一层奖励
        bonus = 1.0
        if is_long and role in {"root", "3rd"}:
            bonus += 0.18
        if is_strong and role in {"root", "3rd", "5th"}:
            bonus += 0.22
        return mass * ROLE_WEIGHTS[role] * bonus, mass

    if chord_pcs and min(_pc_distance(pc, c) for c in chord_pcs) == CLASH_SEMITONE:
        # 小二度/半音撞音：不协和扣分
        return mass * (-0.95), mass

    # 经过音 / 邻音，给很低的中性分
    return mass * 0.08, mass


def calculate_harmony_score(
    melody_notes: Sequence[MelodyNote],
    progression: ChordProgression,
    tempo_bpm: float,
    song_duration: float,
) -> float:
    """
    按和弦时间窗计算旋律与走向的谐和度，返回 0–100。

    内音加权重合：根音最高，其次三音、七音、五音。
    长音与强拍落入内音加分；与和弦音形成小二度则扣分。
    """
    if not melody_notes or song_duration <= 0 or not progression.chords_midi:
        return 0.0

    beat_len = 60.0 / max(tempo_bpm, 1.0)
    chord_len = beat_len * BEATS_PER_CHORD
    n_chords = len(progression.chords_midi)
    role_cache = [chord_tone_roles(voicing) for voicing in progression.chords_midi]

    points = 0.0
    mass = 0.0

    for note in melody_notes:
        t = max(note.start, 0.0)
        note_end = min(note.end, song_duration)
        is_long = note.duration >= LONG_NOTE_SECONDS or note.duration >= beat_len * 1.5

        while t < note_end - 1e-6:
            chord_index = int(t / chord_len) % n_chords
            window_start = math.floor(t / chord_len) * chord_len
            window_end = min(window_start + chord_len, song_duration)
            overlap_end = min(note_end, window_end)
            overlap = overlap_end - t
            if overlap <= 0:
                break

            is_strong = _is_strong_beat(note.start, window_start, beat_len)
            contrib, w = _segment_harmony_points(
                note.pitch,
                overlap,
                is_long,
                is_strong,
                role_cache[chord_index],
            )
            points += contrib
            mass += w
            t = overlap_end

    if mass <= 0:
        return 0.0

    raw = points / mass  # 大约落在 [-0.95, ~1.6]
    normalized = (raw + 0.95) / (1.60 + 0.95) * 100.0
    return float(max(0.0, min(100.0, normalized)))


def get_top_recommendations(
    melody_notes: Sequence[MelodyNote],
    tempo_bpm: float,
    song_duration: float,
    library: Sequence[ChordProgression] | None = None,
    top_n: int = TOP_N,
) -> list[ScoredProgression]:
    """全部走向打分后降序，按 emotion 去重，取最高的 top_n 个不同风格。"""
    library = list(library or CHORD_LIBRARY)
    ranked = sorted(
        (
            ScoredProgression(
                progression=item,
                score=calculate_harmony_score(melody_notes, item, tempo_bpm, song_duration),
            )
            for item in library
        ),
        key=lambda x: x.score,
        reverse=True,
    )

    picked: list[ScoredProgression] = []
    seen_emotions: set[str] = set()
    for item in ranked:
        emotion = item.progression.emotion
        if emotion in seen_emotions:
            continue
        seen_emotions.add(emotion)
        picked.append(item)
        if len(picked) >= top_n:
            break
    return picked


def add_chord_accompaniment(
    midi: pretty_midi.PrettyMIDI,
    progression: ChordProgression,
    tempo_bpm: float,
    duration: float,
) -> None:
    if duration <= 0:
        raise RuntimeError("未从哼唱音频中解析到有效旋律音符。")

    beat_len = 60.0 / max(tempo_bpm, 1.0)
    chord_len = beat_len * BEATS_PER_CHORD
    piano = pretty_midi.Instrument(program=PIANO_PROGRAM, name="Chord Accompaniment")

    start = 0.0
    chord_index = 0
    voicings = progression.chords_midi
    while start < duration:
        end = min(start + chord_len, duration)
        for pitch in voicings[chord_index % len(voicings)]:
            piano.notes.append(
                pretty_midi.Note(
                    velocity=CHORD_VELOCITY,
                    pitch=int(pitch),
                    start=start,
                    end=end,
                )
            )
        chord_index += 1
        start = end

    midi.instruments.append(piano)


def style_piano_tracks(midi: pretty_midi.PrettyMIDI) -> None:
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        instrument.program = PIANO_PROGRAM
        if not instrument.name:
            instrument.name = "Humming Melody"


def clone_midi(midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    return copy.deepcopy(midi)


def sanitize_filename_part(text: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", text).strip()
    return cleaned or "untitled"


def print_recommendations(recs: Sequence[ScoredProgression]) -> None:
    print()
    print("=" * 64)
    print("  和弦走向推荐（乐理谐和度 · 情感去重 Top 4）")
    print("=" * 64)
    for i, rec in enumerate(recs, start=1):
        p = rec.progression
        print(f"  [{i}] {p.emotion}  |  {p.name}  |  匹配 {rec.score:.1f}%")
        print(f"      氛围：{p.vibe_desc}")
        print()
    print("-" * 64)


def prompt_choice(n: int) -> int:
    while True:
        raw = input(f"请输入数字选择方案（1-{n}）：").strip()
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= n:
                return value
        print(f"输入无效，请输入 1 到 {n} 的整数。")


def main() -> None:
    audio_path = Path(__file__).resolve().parent / AUDIO_FILENAME
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到哼唱音频文件：{audio_path}")

    print(f"正在用 basic_pitch 转写旋律：{audio_path.name}")
    _model_output, midi_data, _note_events = predict(str(audio_path))

    melody_notes = extract_melody_notes(midi_data)
    if not melody_notes:
        raise RuntimeError("未从哼唱音频中解析到有效旋律音符。")

    tempo_bpm = estimate_tempo_bpm(midi_data)
    duration = melody_end_time(melody_notes, midi_data)
    print(f"估计速度：{tempo_bpm:.1f} BPM，旋律时长：{duration:.2f}s，开始匹配 {len(CHORD_LIBRARY)} 种和弦走向…")

    recs = get_top_recommendations(melody_notes, tempo_bpm, duration)
    if not recs:
        raise RuntimeError("未能生成和弦推荐。")

    print_recommendations(recs)
    choice = prompt_choice(len(recs))
    selected = recs[choice - 1]
    progression = selected.progression

    song = clone_midi(midi_data)
    style_piano_tracks(song)
    add_chord_accompaniment(song, progression, tempo_bpm, duration)

    filename = f"output_song_{sanitize_filename_part(progression.emotion)}.mid"
    output_path = Path(__file__).resolve().parent / filename
    song.write(str(output_path))

    print()
    print(f"已选择：[{choice}] {progression.emotion} · {progression.name}（{selected.score:.1f}%）")
    print("音色：主旋律与伴奏均为 Acoustic Grand Piano (program=0)")
    print(f"已导出：{output_path}")


if __name__ == "__main__":
    main()
