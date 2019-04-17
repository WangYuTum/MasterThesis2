import sys
from subprocess import call

# create param list
seq_names = ['vot2016_gymnastics2', 'vot2016_fernando', 'vot2016_handball1', 'vot2016_gymnastics1',
             'vot2016_racing', 'vot2016_crossing', 'vot2016_godfather', 'vot2016_marching', 'vot2016_fish1',
             'vot2016_fish4', 'vot2016_gymnastics3', 'vot2016_sheep', 'vot2016_book', 'vot2016_handball2',
             'vot2016_helicopter', 'vot2016_road', 'vot2016_gymnastics4', 'vot2016_nature', 'vot2016_blanket',
             'vot2016_birds1', 'vot2016_soldier', 'vot2016_wiper', 'vot2016_traffic', 'vot2016_fish3', 'vot2016_bag',
             'vot2016_birds2', 'vot2016_iceskater1', 'vot2016_butterfly', 'vot2016_octopus', 'vot2016_ball2',
             'vot2016_fish2', 'vot2016_bmx', 'vot2016_singer3', 'vot2016_sphere', 'vot2016_graduate',
             'vot2016_motocross2', 'vot2016_tunnel', 'vot2016_ball1']

for seq in seq_names:
    arg0 = str(seq)
    call(['python', 'run_tracker.py', arg0])
