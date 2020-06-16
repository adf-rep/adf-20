import math
import random
import copy
import utils.arff as arff


def create_drifting_stream(streams, drift_defs, s_size):
    print('Creating drifting stream')
    drifting_stream = []
    d_idx = 0
    drift = drift_defs[d_idx]
    print(drift)

    for i in range(0, s_size):
        prob = sigm(drift['p'], drift['w'], i)
        r = random.uniform(0, 1)

        if prob > r:
            new_sample = streams[drift['c'][0]][i]
        else:
            new_sample = streams[drift['c'][1]][i]

        drifting_stream.append(new_sample)

        if (i > drift['p'] + drift['w'] / 2.0) and d_idx < len(drift_defs) - 1:
            d_idx += 1
            drift = drift_defs[d_idx]
            print(drift)

    return drifting_stream


def sigm(p, w, x):
    try:
        ans = 1 - (1.0 / (1 + math.exp((-4.0 / w) * (x - p))))
    except OverflowError:
        if x < p:
            ans = 1
        else:
            ans = 0
    return ans


def create_real_drifting_stream(base_stream, drift_defs, mappings, s_size, dbl=False):
    print('Creating drifting stream')
    drifting_stream = []
    d_idx = 0
    drift = drift_defs[d_idx]
    c_idx = 0
    concepts = drift_defs[c_idx]['c']
    dc_num = 0
    first_pw = drift_defs[0]['p'] - (drift_defs[0]['w'] / 2)

    print(drift)
    print(concepts)

    if dbl:
        base_stream = base_stream + base_stream

    for i in range(0, s_size):
        prob = sigm(drift['p'], drift['w'], i)
        r = random.uniform(0, 1)
        base_row = copy.copy(base_stream[i])
        #print(i, base_row)

        if prob > r:
            new_sample_cls = mappings[drift['c'][0]][base_row[-1]]
        else:
            new_sample_cls = mappings[drift['c'][1]][base_row[-1]]

        if new_sample_cls != mappings[concepts[0]][base_row[-1]] and i >= first_pw:
            dc_num += 1

        base_row[-1] = new_sample_cls
        drifting_stream.append(base_row)

        if (i > drift['p'] + drift['w'] / 2.0) and d_idx < len(drift_defs) - 1:
            d_idx += 1
            drift = drift_defs[d_idx]
            print(drift)

        if d_idx >= 1 and i == drift['p'] - drift['w'] / 2.0:
            concepts = drift_defs[d_idx]['c']
            print(i, concepts)

    print(dc_num, s_size - first_pw)
    dc_ratio = dc_num / (s_size - first_pw)
    print(dc_ratio)

    return drifting_stream


def create_dl_drift_streams(root_dir):
    arff_data = arff.load_arff('real-dl/generic/30k/SEMG', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 27000, 'w': 2700, 'c': ['m1', 'm2']}], {
            'm1': {'cyl': 'cyl', 'hook': 'hook', 'lat': 'lat', 'palm': 'palm', 'spher': 'spher', 'tip': 'tip'},
            'm2': {'cyl': 'tip', 'hook': 'cyl', 'lat': 'hook', 'palm': 'lat', 'spher': 'palm', 'tip': 'spher'},
        }, 54000, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/generic/SEMG-D1', root_dir)

    arff_data = arff.load_arff('real-dl/text/30k/AGNEWS-30K', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 30000, 'w': 3000, 'c': ['m1', 'm2']}], {
                                         'm1': {'Business': 'Business', 'SciTech': 'SciTech', 'Sports': 'Sports', 'World': 'World'},
                                         'm2': {'Business': 'World', 'SciTech': 'Business', 'Sports': 'SciTech', 'World': 'Sports'},
        }, 60000, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/text/AGNEWS-D1', root_dir)

    arff_data = arff.load_arff('real-dl/text/30k/BBC', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 33375, 'w': 3300, 'c': ['m1', 'm2']}], {
                                         'm1': {'business': 'business', 'entertainment': 'entertainment', 'politics': 'politics', 'sport': 'sport', 'tech': 'tech'},
                                         'm2': {'business': 'tech', 'entertainment': 'business', 'politics': 'entertainment', 'sport': 'politics', 'tech': 'sport'},
                                     }, 66750, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/text/BBC-D1', root_dir)

    arff_data = arff.load_arff('real-dl/text/30k/SOGOU-30K', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 30000, 'w': 3000, 'c': ['m1', 'm2']}], {
                                         'm1': {'1': '1', '2': '2', '3': '3', '4': '4', '5': '5'},
                                         'm2': {'1': '5', '2': '1', '3': '2', '4': '3', '5': '4'}
                                     }, 60000, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/text/SOGOU-D1', root_dir)

    arff_data = arff.load_arff('real-dl/vis/32/CIFAR10', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 60000, 'w': 6000, 'c': ['m1', 'm2']}], {
                                         'm1': {'airplane': 'airplane', 'automobile': 'automobile', 'bird': 'bird', 'cat': 'cat',
                                                'deer': 'deer', 'dog': 'dog', 'frog': 'frog', 'horse': 'horse', 'ship': 'ship', 'truck': 'truck'},
                                         'm2': {'airplane': 'truck', 'automobile': 'airplane', 'bird': 'automobile', 'cat': 'bird',
                                                'deer': 'cat', 'dog': 'deer', 'frog': 'dog', 'horse': 'frog', 'ship': 'horse', 'truck': 'ship'}
                                     }, 120000, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/vis/CIFAR10-D1', root_dir)

    arff_data = arff.load_arff('real-dl/vis/32/CMATER-BANGLA', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 20000, 'w': 2000, 'c': ['m1', 'm2']}], {
                                         'm1': {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'},
                                         'm2': {'0': '9', '1': '0', '2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '7': '6', '8': '7', '9': '8'}
                                     }, 40000, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/vis/CMATER-BANGLA-D1', root_dir)

    arff_data = arff.load_arff('real-dl/vis/32/DOGS-VS-CATS', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 50000, 'w': 5000, 'c': ['m1', 'm2']}], {
                                         'm1': {'dog': 'dog', 'cat': 'cat'},
                                         'm2': {'dog': 'cat', 'cat': 'dog'}
                                     }, 100000, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/vis/DOGS-VS-CATS-D1', root_dir)

    arff_data = arff.load_arff('real-dl/vis/64/IMAGENETTE', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 18938, 'w': 1800, 'c': ['m1', 'm2']}], {
                                         'm1': {'ball': 'ball', 'church': 'church', 'horn': 'horn', 'parachute': 'parachute',
                                                'player': 'player', 'pump': 'pump', 'saw': 'saw', 'springer': 'springer',
                                                'tench': 'tench', 'truck': 'truck'},
                                         'm2': {'ball': 'truck', 'church': 'ball', 'horn': 'church', 'parachute': 'horn',
                                                'player': 'parachute', 'pump': 'player', 'saw': 'pump', 'springer': 'saw',
                                                'tench': 'springer', 'truck': 'tench'}
                                     }, 37876, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/vis/IMAGENETTE-D1', root_dir)

    arff_data = arff.load_arff('real-dl/vis/32/INTEL-IMGS', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 56136, 'w': 5500, 'c': ['m1', 'm2']}], {
                                         'm1': {'buildings': 'buildings', 'forest': 'forest', 'glacier': 'glacier',
                                                'mountain': 'mountain', 'sea': 'sea', 'street': 'street'},
                                         'm2': {'buildings': 'street', 'forest': 'buildings', 'glacier': 'forest',
                                                'mountain': 'glacier', 'sea': 'mountain', 'street': 'sea'}
                                     }, 112272, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/vis/INTEL-IMGS-D1', root_dir)

    arff_data = arff.load_arff('real-dl/vis/32/MALARIA', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 27558, 'w': 2700, 'c': ['m1', 'm2']}], {
                                         'm1': {'neg': 'neg', 'pos': 'pos'},
                                         'm2': {'neg': 'pos', 'pos': 'neg'}
                                     }, 55116, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/vis/MALARIA-D1', root_dir)

    arff_data = arff.load_arff('real-dl/vis/32/MNIST', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 70000, 'w': 7000, 'c': ['m1', 'm2']}], {
                                         'm1': {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'},
                                         'm2': {'0': '9', '1': '0', '2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '7': '6', '8': '7', '9': '8'}
                                     }, 140000, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/vis/MNIST-D1', root_dir)

    arff_data = arff.load_arff('real-dl/vis/32/MNIST_F', root_dir, False)
    ds = create_real_drifting_stream(arff_data['data'], [
        {'p': 70000, 'w': 7000, 'c': ['m1', 'm2']}], {
                                         'm1': {'Ankle_boots': 'Ankle_boots', 'Bag': 'Bag', 'Coat': 'Coat', 'Dress': 'Dress',
                                                'Pullover': 'Pullover', 'Sandal': 'Sandal', 'Shirt': 'Shirt', 'Sneaker': 'Sneaker',
                                                'T-shirt': 'T-shirt', 'Trouser': 'Trouser'},
                                         'm2': {'Ankle_boots': 'Trouser', 'Bag': 'Ankle_boots', 'Coat': 'Bag', 'Dress': 'Coat',
                                                'Pullover': 'Dress', 'Sandal': 'Pullover', 'Shirt': 'Sandal', 'Sneaker': 'Shirt',
                                                'T-shirt': 'Sneaker', 'Trouser': 'T-shirt'}
                                     }, 140000, True)
    arff.write_arff(arff_data['attributes'], ds, 'real-dl/drift/vis/MNIST_F-D1', root_dir)


def replace_classes(atts, new_classes):
    atts[-1] = ('class', new_classes)
    return atts


def main():
    print("Running...")
    root_dir = '.'

    create_dl_drift_streams(root_dir)


if __name__ == "__main__":
    main()
