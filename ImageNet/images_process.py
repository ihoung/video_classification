import os
import cv2
import argparse

SUPPORTED_FORMATS = ['.jpg', '.png', '.jpeg']


def remove_invalid_images(input_path):
    for filename in os.listdir(input_path):
        ext = os.path.splitext(filename)[1]
        filepath = os.path.join(input_path, filename)
        if (ext not in SUPPORTED_FORMATS) or (cv2.imread(filepath) is None):
            os.system('rm {}'.format(filepath.replace(' ', '\" \"')))
            print('{} is not a valid image file. Delete!'.format(filepath))


def name_exists(dirpath, filename):
    for filetype in SUPPORTED_FORMATS:
        if os.path.exists(os.path.join(dirpath, filename+filetype)):
            return True
    return False


def images_rename(input_path):
    fileset = set()
    dirname = input_path.split(os.sep)[-1]
    filelist = os.listdir(input_path)
    num = len(str(len(filelist)))
    count = 1
    for file in filelist:
        oldfile = os.path.join(input_path, file)
        ext = os.path.splitext(file)[1]
        filename = os.path.splitext(file)[0]
        if (not os.path.isfile(oldfile)) or (filename in fileset):
            continue
        newfilename = dirname + '_' + str(count).zfill(num)
        while name_exists(input_path, newfilename):
            fileset.add(newfilename)
            count += 1
            newfilename = dirname + '_' + str(count).zfill(num)
        os.rename(oldfile, os.path.join(input_path, newfilename+ext))
    print('Images Renaming Done!')


def find_dupes(input_path):
    os.system('fdupes -rdN {}'.format(input_path))


def downscale(input_path, target_edge):
    if target_edge <= 0:
        print('Wrong Size Input!')
        return False
    print('scaning {}...'.format(input_path))
    for filename in os.listdir(input_path):
        filepath = os.path.join(input_path, filename)
        if not os.path.isfile(filepath):
            continue
        img = cv2.imread(filepath)
        if img is None:
            continue
        height, width = img.shape[:2]
        short_edge = min(height, width)
        if short_edge > target_edge:
            scale = float(target_edge) / short_edge
            new_w = int(round(width*scale))
            new_h = int(round(height*scale))
            print('Down sampling {} from {}x{} to {}x{}...'.format(filepath, width, height, new_w, new_h))
            img = cv2.resize(img, (new_w, new_h))
            cv2.imwrite(filepath, img)
    print('Downscaling Done!')
    return True


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Process the images downloaded from the Internet.')
    parse.add_argument('input_path', type=str, help='The folder path of the images being processed.')
    parse.add_argument('--rminval', '-rmi', nargs='?',
                       const=True, type=bool, help='Remove the invalid images.', required=False)
    parse.add_argument('--batchrename', '-brn', const=True, nargs='?',
                       type=bool, help='Batch renaming the images in a folder.', required=False)
    parse.add_argument('--fdupes', '-fd', nargs='?',
                       const=True, type=bool, help='Remove the duplicated images.', required=False)
    parse.add_argument('--downscale', '-ds', nargs=1, type=int, dest='edge',
                       help='Bathc scaling the images to an uniform size.', required=False)
    args = parse.parse_args()
    if args.rminval:
        remove_invalid_images(args.input_path)
    if args.batchrename:
        images_rename(args.input_path)
    if args.fdupes:
        find_dupes(args.input_path)
    if args.edge:
        target_edge = args.edge[0]
        downscale(args.input_path, target_edge)

