import arff
from joblib import Parallel, delayed
import multiprocessing


def load_arffs(fns, root_dir):
    arffs = {}
    num_cores = multiprocessing.cpu_count()
    rs = Parallel(n_jobs=num_cores)(delayed(load_arff)(fn, root_dir) for fn in fns)

    print('Merging')
    for i in range(0, len(rs)):
        key = list(rs[i].keys())[0]
        arffs[key] = rs[i][key]

    return arffs


def load_arff(fp, root_dir, m=True):
    print('Loading {0}'.format(fp))
    name = fp.split('/')[-1]
    arff_data = arff.load(open('{0}/{1}.arff'.format(root_dir, fp), 'r'))

    if m:
        return {name: arff_data}
    else:
        return arff_data


def split_similar_arffs_fields(arffs):
    data = {}
    for key, value in arffs.items():
        data[key] = value['data']

    return [
        data,
        arffs[list(arffs.keys())[0]]['attributes']
    ]


def write_arff(header, data, fp, root_dir):
    print('Writing {0}'.format(fp))
    new_arff = {'attributes': header, 'data': data, 'relation': fp, 'description': ''}

    with open('{0}/{1}.arff'.format(root_dir, fp), "w") as fh:
        arff.dump(new_arff, fh)
