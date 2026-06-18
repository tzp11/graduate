from pathlib import Path

from research.reliability.datasets.dior import _coco_objects_to_yolo, convert_dior_split, write_dataset_yaml


def test_dior_xml_is_converted_to_yolo_labels(tmp_path: Path):
    images = tmp_path / "source_images"
    annotations = tmp_path / "source_annotations"
    images.mkdir()
    annotations.mkdir()
    (images / "a.jpg").write_bytes(b"placeholder")
    (annotations / "a.xml").write_text(
        """<annotation><size><width>100</width><height>200</height></size>
<object><name>ship</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox></object>
</annotation>""",
        encoding="ascii",
    )
    output = tmp_path / "converted"
    assert convert_dior_split(images_dir=images, annotations_dir=annotations, output_root=output, split="train") == 1
    line = (output / "labels" / "train" / "a.txt").read_text(encoding="ascii").strip()
    assert line == "13 0.20000000 0.20000000 0.20000000 0.20000000"
    assert write_dataset_yaml(output).exists()


def test_huggingface_coco_boxes_are_converted_to_yolo_labels():
    objects = {"category": [13], "bbox": [[10, 20, 20, 40]]}
    assert _coco_objects_to_yolo(objects, width=100, height=200) == [
        "13 0.20000000 0.20000000 0.20000000 0.20000000"
    ]
