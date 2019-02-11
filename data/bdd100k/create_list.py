import glob
import os
import random
from xml.etree.ElementTree import parse


def basename_(filename):
    return os.path.splitext(os.path.basename(filename))[0]


if __name__ == "__main__":
    save_dir = os.getcwd()
    root_dir = '{}/data/BDD/bdd100k'.format(os.getenv("HOME"))
    os.chdir(root_dir)
    img_dir = 'images/100k'
    xml_dir = 'xml'
    sub_dir = {'train': 'train', 'val': 'val'}

    for dataset in ['train', 'val']:
        dst_file = "{}/{}.txt".format(save_dir, dataset)

        xml_list = glob.glob(os.path.join(xml_dir, sub_dir[dataset], '*.xml'))
        xml_list.sort()

        with open(dst_file, 'w') as f:
            for xml in xml_list:
                img = xml.replace('.xml', '.jpg').replace(xml_dir, img_dir)
                assert basename_(img) == basename_(xml), "img: %s, xml: %s" % (basename_(img), basename_(xml))
                line = img + ' ' + xml + '\n'
                f.write(line)

        if dataset in ['train']:
            with open(dst_file) as f:
                lines = f.readlines()

            random.shuffle(lines)
            with open(dst_file, 'w') as f:
                for line in lines:
                    f.write(line)

        # create name size txt file
        if dataset in ['val']:
            dst_file = "{}/{}_name_size.txt".format(save_dir, dataset)
            with open(dst_file, 'w') as f:
                for xml in xml_list:
                    tree = parse(xml)
                    note = tree.getroot()
                    img_filename = os.path.splitext(note.findtext("filename"))[0]
                    width = note.find("size").findtext("width")
                    height = note.find("size").findtext('height')

                    line = img_filename + ' ' + width + ' ' + height + '\n'
                    f.write(line)
