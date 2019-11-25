import os
import subprocess
import shutil

folders = 'hyper_param_runs/res/'
dest = 'hyper_param_runs/res/plot/'
dest_file = 'plot.png'

experiment = 'cifar_si'
approach = 'alexnet_si'

default_sgd = 'sgd'
default_random = 'random'
default_joint = 'joint'

new_approaches = {}
new_files = {}

entries = os.listdir(folders)
for entry in entries:
    if os.path.isdir(os.path.join(folders, entry)):
        continue
    e = os.path.splitext(entry)[0].replace(approach, '')
    e = e.replace(experiment, '')
    e = e.replace('__', '')
    e = e.replace('0_', '')

    new_approaches[entry] = approach+'_'+e
    new_files[entry] = experiment+'_'+approach+'_'+e+'_0.txt'

if not os.path.exists(dest):
    os.makedirs(dest)
for entry in entries:
    if os.path.isdir(os.path.join(folders, entry)):
        continue
    fn = dest+new_files[entry]
    if os.path.exists(fn):
        os.remove(fn)
    try:
        shutil.copy(folders+entry, fn)
        print('done: copy(' + folders+entry + ', ' + fn+')')
    except:
        print('skipped: copy(' + folders+entry + ', ' + fn+')')

# PARSE arguments
approaches = ''

# add defaults
approaches += default_sgd
approaches += ','+default_random
approaches += ','+default_joint

approaches += ''.join([',%s' % a for _,a in new_approaches.items()])

print(5*'\n')
exec_str = ' '.join(['./plot_results.py', '--folders', dest, '--experiment', experiment, '--output', dest+dest_file, '--seeds', '0', '--approaches', approaches])
print(exec_str)