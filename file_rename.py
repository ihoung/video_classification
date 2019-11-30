import os
import sys


def filerename(path, count=1):
    dirname = path.split(os.sep)[-1]
    filelist = os.listdir(path)
    num = len(str(len(filelist)))
    filelist.sort()
    for file in filelist:
        olddir = os.path.join(path, file)
        if not os.path.isfile(olddir):
            continue
        filetype = os.path.splitext(file)[1]
        newdir = os.path.join(path, dirname+'_'+str(count).zfill(num)+filetype)
        while os.path.exists(newdir):
            count += 1
            newdir = os.path.join(path, dirname + '_' + str(count).zfill(num) + filetype)
        os.rename(olddir, newdir)
        count += 1


if __name__ == '__main__':
    filepath = os.path.join(os.getcwd(), sys.argv[1])
    filerename(filepath)
