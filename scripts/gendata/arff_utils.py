

def image_stream_to_arff(xy, size, name, out_path):
    height, width = size
    f = open(out_path, 'w')
    f.write(f'@relation \'{name}_{height}x{width}\'\n\n')

    for i in range(height):
        for j in range(width):
            f.write(f'@attribute coord_{i+1}x{j+1} numeric\n')

    write_labels_and_data(f, xy)
    f.close()


def text_stream_to_arff(xy, size, name, out_path):
    f = open(out_path, 'w')
    f.write(f'@relation \'{name}_{size}\'\n\n')

    for i in range(size):
        f.write(f'@attribute word_{i+1} integer\n')

    write_labels_and_data(f, xy)
    f.close()


def generic_stream_to_arff(xy, size, name, out_path):
    f = open(out_path, 'w')
    f.write(f'@relation \'{name}_{size}\'\n\n')

    for i in range(size):
        f.write(f'@attribute sample_{i+1} numeric\n')

    write_labels_and_data(f, xy)
    f.close()


def write_labels_and_data(f, xy):
    labels = set()
    for row in xy:
        labels.add(label_str_conv(row[-1]))
    labels = list(labels)
    labels.sort()

    labels_str = ','.join(labels)
    f.write(f'@attribute class {{{labels_str}}}\n')

    f.write('\n@data\n\n')
    for row in xy:
        f.write(','.join(str(x) for x in row[:-1]) + ',' + label_str_conv(row[-1]) + '\n')


def label_str_conv(label):
    return str(int(label)) if isinstance(label, float) else str(label)
