import csv
import io
import unittest
import zipfile

from PIL import Image

from prepare_dataset import Record, inspect_records, source_group, split_groups


class DatasetSplitTests(unittest.TestCase):
    def test_adjacent_frames_and_acquisition_repeats_share_group(self):
        first = "R1-1100um-top-to-bottom-center_frame_85_patch_1.png"
        repeat = "R1-1100um-top-to-bottom-center-1_frame_220_patch_7.png"
        self.assertEqual(source_group(first), source_group(repeat))
        with self.assertRaises(ValueError):
            source_group("image_1.png")

    def test_four_splits_and_deterministic_group_assignment(self):
        records = [
            Record(str(index), str(group), "", "", "", index % 2 == 0, "", "")
            for group in range(7) for index in range(10)
        ]
        assigned = split_groups(records, 2026)
        self.assertEqual(assigned, split_groups(list(reversed(records)), 2026))
        self.assertEqual(set(assigned.values()), {"train", "val", "calibration", "test"})
        self.assertEqual(len(assigned), 7)

    def test_same_decoded_image_is_deduplicated_and_conflicting_masks_are_excluded(self):
        data = io.BytesIO()
        meta = io.StringIO()
        writer = csv.writer(meta)
        writer.writerow(["file_name", "original_file_name"])
        with zipfile.ZipFile(data, "w") as z:
            for idx, (intensity, red) in enumerate([(64, 255), (64, 255), (192, 0), (192, 255)], 1):
                stem = f"image_{idx}"
                writer.writerow([stem + ".jpg", f"R1-700um-top_frame_{idx}_patch_1.png"])
                pic, mask = io.BytesIO(), io.BytesIO()
                Image.new("RGB", (8, 8), (intensity,) * 3).save(pic, format="JPEG")
                Image.new("RGBA", (8, 8), (red, 0, 0, 0)).save(mask, format="PNG")
                z.writestr(f"segmentation/images/{stem}.jpg", pic.getvalue())
                z.writestr(f"segmentation/masks/{stem}.png", mask.getvalue())
            z.writestr("classification/labels.csv", meta.getvalue())
        with zipfile.ZipFile(data) as z:
            unique, stats = inspect_records(z)
        self.assertEqual([(r.name, r.positive) for r in unique], [("image_1", True)])
        self.assertEqual(stats["exact_pixel_duplicates_removed"], 1)
        self.assertEqual(stats["conflicting_duplicate_records_excluded"], ["image_3", "image_4"])


if __name__ == "__main__":
    unittest.main()
