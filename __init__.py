import sys, glob, os
sys.path.append('deep-learning-base')

for p in glob.glob('deep-learning-base/*'):
    if os.path.isdir(p):
        sys.path.append(p)



DATA_PATH_IMAGENET = '/NS/twitter_archive/work/vnanda/data'
DATA_PATH = '/NS/robustness_4/work/vnanda/data'
DATA_PATH_FLOWERS_PETS = '/NS/robustness_1/work/vnanda/data'

SERVER_PROJECT_PATH = 'partially_inverted_reps'