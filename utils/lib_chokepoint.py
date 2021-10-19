import os

def get_all_in_dir(parent, extension=None):
    all = []
    if extension != None:
        all = [os.path.join(parent, child) for child in os.listdir(parent) if child.endswith(extension)]
    else:
        all = [os.path.join(parent, child) for child in os.listdir(parent)]
    all.sort()
    return all

def load_images_chokepoint(images_dir):
    image_paths = []
    pdirs = [os.path.join(images_dir, pdir) for pdir in os.listdir(images_dir)]
    pdirs.sort()
    for pdir in pdirs:
        cdirs = get_all_in_dir(pdir)
        has_sub_dir = check_has_sub_dir_chokepoint(pdir)
        for cdir in cdirs:
            if has_sub_dir:
                csdirs = get_all_in_dir(cdir)
                for csdir in csdirs:
                    imps = get_all_in_dir(csdir, '.jpg')
                    image_paths.append(imps)
            else:
                imps = get_all_in_dir(cdir, '.jpg')
                image_paths.append(imps)
    return image_paths

def check_has_sub_dir_chokepoint(pdir):
    has_sub_dir = False
    name = pdir.rsplit('/', 1)[-1].split('_')
    if name[0] == 'P2E' or name[0] == 'P2L':
        if name[1] != 'S5':
            has_sub_dir = True
    return has_sub_dir