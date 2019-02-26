'''
 Generate a file for each snippet (train/val), the file contains all possible track_ids and the
 corresponding image lists. The file structure starting from line0:

 track_id0
 frame_id0 img_path0 xmax xmin ymax ymin width height
 frame_id1 img_path1 xmax xmin ymax ymin width height
 ...
 ...
 ...
 #####
 track_idn
 frame_id0 img_path0 xmax xmin ymax ymin width height
 frame_id1 img_path1 xmax xmin ymax ymin width height
 #####


 track_idx: unique within each snippet
 frame_idx: the frame id within the current snippet
 img_pathx: the image that contains the track_id
 bbox(xmax xmin ymax ymin): the bbox of the track_id in this image
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import xml.etree.ElementTree as ET

SNIP_ID_FILE = '../../tmp_data/imgnet-vid/snip_id_frames.txt'
SAVE_FILE_DIR = '../../tmp_data/imgnet-vid/snippet_objects/'


def parse_xml(xml_file):
    '''
    :param xml_file:
    :return: parsed data
    '''

    try:
        tree = ET.parse(xml_file)
    except Exception:
        print('Failed to parse: {}'.format(xml_file))
        sys.exit(-1)
    root = tree.getroot()

    # parse
    parsed_data = []
    object_list = root.findall('object')
    if len(object_list) == 0: # if no object in current frame
        parsed_data.append(-1)
        return parsed_data
    else:
        # get image height, width
        width = int(root.find('size').find('width').text)
        parsed_data.append(width)
        height = int(root.find('size').find('height').text)
        parsed_data.append(height)
        # get each object track_id, bbox
        for obj in object_list:
            track_id = int(obj.find('trackid').text) # object id: int
            bbox = obj.find('bndbox')
            xmax = int(bbox.find('xmax').text)
            xmin = int(bbox.find('xmin').text)
            ymax = int(bbox.find('ymax').text)
            ymin = int(bbox.find('ymin').text)
            parsed_data.append({'track_id': track_id, 'xmax':xmax, 'xmin':xmin, 'ymax':ymax, 'ymin':ymin})

    return parsed_data

def main():

    # read snippet ids and dirs
    with open(SNIP_ID_FILE) as f:
        content = f.read().splitlines()
    sorted(content)
    print('Number of snippets: {}'.format(len(content)))
    snippet_list = [ [int(item.split(' ')[0]), int(item.split(' ')[2]), item.split(' ')[1]] for item in content]
    if len(snippet_list) != len(content):
        raise ValueError('Length error')

    # each item in snippet_list is a list: [snip_id, num_frames image_dir]
    for item in snippet_list:
        snippet_id = item[0]
        # make a file
        print('Writing snippet: {}'.format(snippet_id))
        f = open(os.path.join(SAVE_FILE_DIR, 'snippet_%d'%snippet_id), 'w')
        # get current image/annotation dir
        num_frames = item[1]
        img_dir = item[2]
        anno_dir = img_dir.replace('Data', 'Annotations')
        # get image_list and anno_list
        image_list = sorted([os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)])
        anno_list = sorted([os.path.join(anno_dir, anno_name) for anno_name in os.listdir(anno_dir)])
        if len(image_list) != len(anno_list) or len(image_list) != num_frames:
            raise ValueError('Number images != Number of annotations: {}, {}'.format(img_dir, anno_dir))

        # parse each xml
        track_dict = {} # indexed by track_id
        for frame_id, anno in enumerate(anno_list):
            img_path = image_list[frame_id] # the corresponding image
            parsed_data = parse_xml(anno)
            if parsed_data[0] == -1:
                print('No object in frame: {}'.format(anno))
                continue
            else:
                width = parsed_data[0]
                height = parsed_data[1]
                # for each object in current frame
                for bbox in parsed_data[2:]:
                    track_id = bbox['track_id']
                    pack_data = {'frame_id':frame_id, 'img_path':img_path, 'bbox': bbox, 'width':width, 'height':height}
                    if track_id in track_dict: # if track_id is already in the dict, append data
                        track_dict[track_id].append(pack_data)
                    else:
                        track_dict[track_id] = [] # create a list this track_id
                        track_dict[track_id].append(pack_data)
        # parse all xml done

        # get all track_ids and write to file
        for track_id in track_dict:
            # each value is a list
            f.write(str(track_id) + '\n') # write track_id
            for pack_data in track_dict[track_id]:
                # write data: frame_id img_path xmax xmin ymax ymin width height
                f.write(str(pack_data['frame_id']) + ' '
                        + pack_data['img_path'] + ' '
                        + str(pack_data['bbox']['xmax']) + ' ' + str(pack_data['bbox']['xmin']) + ' '
                        + str(pack_data['bbox']['ymax']) + ' ' + str(pack_data['bbox']['ymin']) + ' '
                        + str(pack_data['width']) + ' ' + str(pack_data['height']) + '\n')
            f.write('#####' + '\n') # mark the end of current track_id
        # write and close file
        f.flush()
        f.close()

if __name__ == '__main__':
    main()