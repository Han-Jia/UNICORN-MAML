import os

f_list = ['train', 'val', 'test']
for name in f_list:
    with open(name + '.csv', 'w') as f:
        f.write('filename,label\n')
        dirlist = os.listdir(os.path.join('.', name))
        print('Set {} Class Number: {}'.format(name, len(dirlist)))
        for d_name in dirlist:
            file_list = os.listdir(os.path.join('.', name, d_name))
            for file in file_list:
                f.write(','.join(['/'.join([name, d_name, file]), d_name]) + '\n')