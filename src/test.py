"""
Quick test để kiểm tra post-processor hoạt động
"""

from post_processing import TextPostProcessor

# Test cases từ output thực tế của bạn
test_cases = [
    # (raw_output, expected_label)
    ("te is also a diredor s a couple opgrages , tno he mdi timeas wel to bea brie",
     "He is also a director of a couple of garages . And he finds time as well to be a lyric"),

    ("The yumbers inelude \" Seothand the erave 's atlen of Marken , s",
     "The numbers include \" Scotland the Brave , \" \" Men of Harlech , \""),

    ("which is most likdy to have been the original",
     "which is most likely to have been the original"),

    ("Bospel represents the worl of generations of",
     "Gospel represents the work of generations of"),

    ("Potter soreamed daring au ectiou pund was omestod , the",
     "Potter screamed during an action , and was arrested . He"),

    ("Northern d etiome rthe cameras ployed continnousby ow his",
     "Northern defiance . The cameras played continuously on his"),
]

processor = TextPostProcessor()

print("=" * 80)
print("POST-PROCESSOR TEST")
print("=" * 80)

for i, (raw, expected) in enumerate(test_cases, 1):
    print(f"\n[Test {i}]")
    print(f"Raw:      {raw}")
    print(f"Expected: {expected}")

    # Safe mode
    safe = processor.process(raw, aggressive=False)
    print(f"Safe:     {safe}")

    # Aggressive mode
    agg = processor.process(raw, aggressive=True)
    print(f"Aggro:    {agg}")

    # Simple comparison
    raw_words = set(raw.lower().split())
    expected_words = set(expected.lower().split())
    safe_words = set(safe.lower().split())
    agg_words = set(agg.lower().split())

    raw_overlap = len(raw_words & expected_words) / len(expected_words) * 100
    safe_overlap = len(safe_words & expected_words) / len(expected_words) * 100
    agg_overlap = len(agg_words & expected_words) / len(expected_words) * 100

    print(f"\nWord overlap with expected:")
    print(f"  Raw:  {raw_overlap:.1f}%")
    print(f"  Safe: {safe_overlap:.1f}%")
    print(f"  Agg:  {agg_overlap:.1f}%")
    print("-" * 80)