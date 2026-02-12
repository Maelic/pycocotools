# pycocotools for Visual Relationships Detection & Scene Graph Generation

This is a fork of the [ppwwyyxx version fot he coco API](https://github.com/ppwwyyxx/cocoapi), with the support of relationships annotations. The aim is to provide a standardize way of loading visual relationships from different datasets (Visual Genome, PSG etc). The visual relationships annotations can be included in a coco-format .json file (with box annotations) as follows:

```json
{
  "rel_annotations": [
    {
      "id": 0,
      "subject_id": 1,
      "predicate_id": 1,
      "object_id": 2,
      "image_id": 1
    }
  ],
  "rel_categories": [
    {"id": 0, "name": "riding",  "supercategory": "action"},
    {"id": 1, "name": "under", "supercategory": "spatial"}
  ]
}
```

Please see the [pycocoDemo.ipynb](pycocoDemo.ipynb) file for more information on how to load the relationship data.

## Quick Install:
```
pip install git+https://github.com/Maelic/pycocotools
```

## Local build:
```
pip install .
```

## Instructions for maintainers: to build a sdist package:
```
python -m build --sdist .
```

### TODO: Add the Eval Code for Relationships (simple implementation of the Recall@K and meanRecall@K)