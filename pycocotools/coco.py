__author__ = 'tylin'
__version__ = '2.0'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  getRelIds  - Get relationship ids that satisfy given filter conditions.
#  getRelCatIds - Get relationship category ids that satisfy given filter conditions.
#  loadRels   - Load relationship annotations with the specified ids.
#  loadRelCats - Load relationship categories with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations and relationships.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, "img"=image, and "rel"=relationship.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>getRelIds, COCO>getRelCatIds,
# COCO>loadRels, COCO>loadRelCats, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import numpy as np
import copy
import itertools
from . import mask as maskUtils
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        # relationship data structures
        self.rels, self.relCats = dict(), dict()
        self.imgToRels, self.relCatToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        # relationship indices
        rels, relCats = {}, {}
        imgToRels, relCatToImgs = defaultdict(list), defaultdict(list)
        
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        # handle relationship annotations
        if 'rel_annotations' in self.dataset:
            for rel in self.dataset['rel_annotations']:
                imgToRels[rel['image_id']].append(rel)
                rels[rel['id']] = rel

        if 'rel_categories' in self.dataset:
            for relCat in self.dataset['rel_categories']:
                relCats[relCat['id']] = relCat

        if 'rel_annotations' in self.dataset and 'rel_categories' in self.dataset:
            for rel in self.dataset['rel_annotations']:
                relCatToImgs[rel['predicate_id']].append(rel['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        # relationship class members
        self.rels = rels
        self.imgToRels = imgToRels
        self.relCats = relCats
        self.relCatToImgs = relCatToImgs

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def getRelIds(self, imgIds=[], relCatIds=[], subjCatIds=[], objCatIds=[]):
        """
        Get relationship ids that satisfy given filter conditions.
        :param imgIds (int array)      : get rels for given imgs
               relCatIds (int array)   : get rels for given predicate categories
               subjCatIds (int array)  : get rels for given subject categories
               objCatIds (int array)   : get rels for given object categories
        :return: ids (int array)       : integer array of rel ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        relCatIds = relCatIds if _isArrayLike(relCatIds) else [relCatIds]
        subjCatIds = subjCatIds if _isArrayLike(subjCatIds) else [subjCatIds]
        objCatIds = objCatIds if _isArrayLike(objCatIds) else [objCatIds]

        if len(imgIds) == len(relCatIds) == len(subjCatIds) == len(objCatIds) == 0:
            rels = self.dataset.get('rel_annotations', [])
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToRels[imgId] for imgId in imgIds if imgId in self.imgToRels]
                rels = list(itertools.chain.from_iterable(lists))
            else:
                rels = self.dataset.get('rel_annotations', [])
            
            # Filter by predicate category
            rels = rels if len(relCatIds) == 0 else [rel for rel in rels if rel['predicate_id'] in relCatIds]
            
            # Filter by subject and object categories
            if len(subjCatIds) > 0 or len(objCatIds) > 0:
                filtered_rels = []
                for rel in rels:
                    # Get subject and object annotations
                    subj_ann = self.anns.get(rel['subject_id'])
                    obj_ann = self.anns.get(rel['object_id'])
                    
                    if subj_ann and obj_ann:
                        subj_cat_match = len(subjCatIds) == 0 or subj_ann['category_id'] in subjCatIds
                        obj_cat_match = len(objCatIds) == 0 or obj_ann['category_id'] in objCatIds
                        
                        if subj_cat_match and obj_cat_match:
                            filtered_rels.append(rel)
                rels = filtered_rels
        
        ids = [rel['id'] for rel in rels]
        return ids

    def getRelCatIds(self, relCatNms=[], supNms=[], relCatIds=[]):
        """
        Get relationship category ids that satisfy given filter conditions.
        :param relCatNms (str array)  : get rel cats for given rel cat names
        :param supNms (str array)     : get rel cats for given supercategory names
        :param relCatIds (int array)  : get rel cats for given rel cat ids
        :return: ids (int array)      : integer array of rel cat ids
        """
        relCatNms = relCatNms if _isArrayLike(relCatNms) else [relCatNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        relCatIds = relCatIds if _isArrayLike(relCatIds) else [relCatIds]

        if len(relCatNms) == len(supNms) == len(relCatIds) == 0:
            relCats = self.dataset.get('rel_categories', [])
        else:
            relCats = self.dataset.get('rel_categories', [])
            relCats = relCats if len(relCatNms) == 0 else [cat for cat in relCats if cat['name'] in relCatNms]
            relCats = relCats if len(supNms) == 0 else [cat for cat in relCats if cat['supercategory'] in supNms]
            relCats = relCats if len(relCatIds) == 0 else [cat for cat in relCats if cat['id'] in relCatIds]
        ids = [cat['id'] for cat in relCats]
        return ids

    def loadRels(self, ids=[]):
        """
        Load relationship annotations with the specified ids.
        :param ids (int array)       : integer ids specifying rels
        :return: rels (object array) : loaded rel objects
        """
        if _isArrayLike(ids):
            return [self.rels[id] for id in ids]
        elif type(ids) == int:
            return [self.rels[ids]]

    def loadRelCats(self, ids=[]):
        """
        Load relationship categories with the specified ids.
        :param ids (int array)           : integer ids specifying rel cats
        :return: relCats (object array)  : loaded rel cat objects
        """
        if _isArrayLike(ids):
            return [self.relCats[id] for id in ids]
        elif type(ids) == int:
            return [self.relCats[ids]]

    def showAnns(self, anns, draw_bbox=False, rels=None, draw_relations=True):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :param draw_bbox (bool): whether to draw bounding boxes
        :param rels (array of object): relationship annotations to display
        :param draw_relations (bool): whether to draw relationship lines and labels
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Polygon

            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            # Store annotation centers for relationship drawing
            ann_centers = {}
            
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                
                # Calculate annotation center for relationships
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    center_x = bbox[0] + bbox[2] / 2
                    center_y = bbox[1] + bbox[3] / 2
                    ann_centers[ann['id']] = (center_x, center_y)
                
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                            # If no bbox, use polygon centroid
                            if ann['id'] not in ann_centers:
                                ann_centers[ann['id']] = (poly[:, 0].mean(), poly[:, 1].mean())
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))
                        # Use mask centroid if no bbox
                        if ann['id'] not in ann_centers:
                            y_coords, x_coords = np.where(m > 0)
                            if len(x_coords) > 0:
                                ann_centers[ann['id']] = (x_coords.mean(), y_coords.mean())
                                
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk]>0):
                            plt.plot(x[sk],y[sk], linewidth=3, color=c)
                    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
                    # Use keypoints centroid if no bbox
                    if ann['id'] not in ann_centers:
                        valid_x = x[v > 0]
                        valid_y = y[v > 0]
                        if len(valid_x) > 0:
                            ann_centers[ann['id']] = (valid_x.mean(), valid_y.mean())

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4,2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

                    # add the label for the bbox on bottom left inside a text box of the same color as the bbox
                    ax.text(bbox_x, bbox_y, self.cats[ann['category_id']]['name'], fontsize=10, color='white',
                            bbox=dict(facecolor=c, alpha=0.7, pad=2, edgecolor='none', boxstyle='round,pad=0.2'))
                    
                    
                    

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
            
            # Draw relationships
            if rels is not None and draw_relations and len(rels) > 0:
                for rel in rels:
                    subj_id = rel['subject_id']
                    obj_id = rel['object_id']
                    predicate_id = rel['predicate_id']
                    
                    # Get centers of subject and object annotations
                    if subj_id in ann_centers and obj_id in ann_centers:
                        subj_center = ann_centers[subj_id]
                        obj_center = ann_centers[obj_id]
                        
                        # Draw arrow from subject to object
                        dx = obj_center[0] - subj_center[0]
                        dy = obj_center[1] - subj_center[1]
                        
                        # Shorten arrow to avoid overlapping with objects
                        arrow_scale = 0.8
                        start_x = subj_center[0] + dx * (1 - arrow_scale) / 2
                        start_y = subj_center[1] + dy * (1 - arrow_scale) / 2
                        end_x = obj_center[0] - dx * (1 - arrow_scale) / 2
                        end_y = obj_center[1] - dy * (1 - arrow_scale) / 2
                        
                        # Draw arrow
                        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.8))
                        
                        # Add relationship label
                        if predicate_id in self.relCats:
                            rel_name = self.relCats[predicate_id]['name']
                            mid_x = (start_x + end_x) / 2
                            mid_y = (start_y + end_y) / 2
                            ax.text(mid_x, mid_y, rel_name, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                                   fontsize=8, ha='center', va='center')
                        
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['info'] = copy.deepcopy(self.dataset.get('info', {}))
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile).__name__ == 'unicode'):
            with open(resFile) as f:
                data = json.load(f)
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
            data = anns
        else:
            data = resFile
        
        # Check if data is a dict (containing both annotations and relationships) or just a list
        if isinstance(data, dict):
            anns = data.get('annotations', [])
            rels = data.get('rel_annotations', [])
            rel_cats = data.get('rel_categories', [])
        else:
            anns = data if isinstance(data, list) else []
            rels = []
            rel_cats = []
        
        # Handle regular annotations
        if len(anns) > 0:
            assert type(anns) == list, 'results annotations are not an array of objects'
            annsImgIds = [ann['image_id'] for ann in anns]
            assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
                   'Results do not correspond to current coco set'
            if 'caption' in anns[0]:
                imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
                res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
                for id, ann in enumerate(anns):
                    ann['id'] = id+1
            elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
                res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                for id, ann in enumerate(anns):
                    bb = ann['bbox']
                    x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                    if not 'segmentation' in ann:
                        ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                    ann['area'] = bb[2]*bb[3]
                    ann['id'] = id+1
                    ann['iscrowd'] = 0
            elif 'segmentation' in anns[0]:
                res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                for id, ann in enumerate(anns):
                    # now only support compressed RLE format as segmentation results
                    ann['area'] = maskUtils.area(ann['segmentation'])
                    if not 'bbox' in ann:
                        ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                    ann['id'] = id+1
                    ann['iscrowd'] = 0
            elif 'keypoints' in anns[0]:
                res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                for id, ann in enumerate(anns):
                    s = ann['keypoints']
                    x = s[0::3]
                    y = s[1::3]
                    x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                    ann['area'] = (x1-x0)*(y1-y0)
                    ann['id'] = id + 1
                    ann['bbox'] = [x0,y0,x1-x0,y1-y0]
            
            res.dataset['annotations'] = anns
        
        # Handle relationship annotations
        if len(rels) > 0:
            # Copy relationship categories from original dataset or use provided ones
            if len(rel_cats) > 0:
                res.dataset['rel_categories'] = rel_cats
            elif 'rel_categories' in self.dataset:
                res.dataset['rel_categories'] = copy.deepcopy(self.dataset['rel_categories'])
            
            # Validate relationship annotations
            for id, rel in enumerate(rels):
                if 'id' not in rel:
                    rel['id'] = id + 1
                # Ensure required fields are present
                assert 'subject_id' in rel, 'Relationship annotation missing subject_id'
                assert 'object_id' in rel, 'Relationship annotation missing object_id'
                assert 'predicate_id' in rel, 'Relationship annotation missing predicate_id'
                assert 'image_id' in rel, 'Relationship annotation missing image_id'
            
            res.dataset['rel_annotations'] = rels
        
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.createIndex()
        return res

    def download(self, tarDir = None, imgIds = [] ):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        if tarDir is None:
            print('Please specify target directory')
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urlretrieve(img['coco_url'], fname)
            print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time()- tic))

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m
