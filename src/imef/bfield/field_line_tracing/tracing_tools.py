from geopack import geopack
import numpy as np
import json
from os.path import exists
from pathlib import Path
import requests
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle


def download_g_file(filepath):
   quin_url = 'https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/'
   filename = filepath.split('/')[-1]
   filedate = filename.split('_')[1]
   fileyear = filedate[0:4]

   remote_location = quin_url + fileyear + '/' + filename

   filepath_path = Path(filepath)
   if not filepath_path.parent.exists():
      filepath_path.parent.mkdir(parents=True)

   # I SHOULD VERIFY THIS, BUT FOR SOME REASON I GET AN ERROR EVERYTIME I TRY TO ACCESS THE SITE, EVEN IN BROWSER
   # SOME DAYS IT WORKS, BUT MOST IT DONT.
   r = requests.get(remote_location, verify=False)

   # Write file to local path
   with open(filepath, 'wb') as file:
      file.write(r.content)

   print('done!')


def find_dict_in_list(lst, key, val):
   for i, dic in enumerate(lst):
      if dic[key] == val:
         return i
   return 'Something has gone terribly wrong'


def get_g_file(timestamp):
   dirstr = 'QinDenton/'
   dirstr = dirstr + timestamp[0:4] + '/QinDenton_' + timestamp[0:8] + '_1min.txt'
   g_file = os.path.join(os.getcwd(), dirstr)
   return g_file


def get_g_params(time):
   time_string=datetimestr(time)
   g_file = get_g_file(time_string)
   g_list = read_qindenton_json(g_file)  # need to find the line for the given timestamp
   ts = str(time_string)
   ts = ts[0:4] + '-' + ts[4:6] + '-' + ts[6:8] + 'T' + ts[8:10] + ':' + ts[10:12] + ':00'
   g_data = g_list[find_dict_in_list(g_list, 'DateTime', ts)]
   g_data = [
      float(g_data['Pdyn']),
      float(g_data['Dst']),
      float(g_data['ByIMF']),
      float(g_data['BzIMF']),
      float(g_data['G']['G1']),
      float(g_data['G']['G2']),
   ]
   return g_data


def read_qindenton_json(filename):
   if exists(filename) == 0:
      download_g_file(filename)

   with open(filename, "r") as f:

      jstring = ""
      for l in f:
         if l.startswith("#"):
            if l.endswith("End JSON\n"):
               jstring += "}"
               break
            else:
               jstring += l[1:]

      preprocessed = []
      for l in f:
         preprocessed.append(l.split())    # fill preprocessed[] with a list of lists (of values) for every line in a file
            
      j = json.loads(jstring)             # convert big file header string into a json object
      realkeys = list(j.keys())[1:]       # top-level keys (e.g., "G" (not "G1" "G2" or "G3"))

      data = [{} for _ in preprocessed]

      for key in realkeys:
         # single column
         if not "DIMENSION" in j[key]:
            column = int(j[key]["START_COLUMN"])
            for i in range(len(data)):
               data[i][key] = preprocessed[i][column]
         # multi-dimensional
         else:
            column = int(j[key]["START_COLUMN"])
            elements = j[key]["ELEMENT_NAMES"]
            #print(elements)
            for i in range(len(data)):
               data[i][key] = {}
               for k, e in enumerate(elements):
                  data[i][key][e] = preprocessed[i][column + k]

   return data


def datetimestr(map_time):
   full = str(map_time)
   dtstr = full[0:4] + full[5:7] + full[8:10] + full[11:13] + full[14:16]
   return dtstr  # 'YYYYMMDDHHMM'


def get_epoch(dt):
   seconds = (dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
   return seconds


def recalc(time, v_drift=None):  # chance that this needs to be a required variable
   timestamp = get_epoch(time)
   if v_drift is None:
      ps = geopack.recalc(timestamp)
   else:
      ps = geopack.recalc(timestamp, v_drift[0], v_drift[1], v_drift[2])


def predict_b_gsm(time, GSM_coords, v_drift):
   # note that coords must be in GSM, but v_drift must be in GSE (solar wind velocity is convective velocity right?)
   # I have to figure out how to convert L, MLT, MLAT in GSE to cartesian GSM coords. Can also do GSW coords if thats easier. For now use example point

   recalc(time, v_drift)

   bxgsm, bygsm, bzgsm = geopack.dip(GSM_coords[0], GSM_coords[1], GSM_coords[2])

   # convert back to gse?
   return np.array([bxgsm, bygsm, bzgsm])


def dual_half_circle(center=(0, 0), radius=1, angle=90, ax=None, colors=('w', 'k', 'k'),
                     **kwargs):
   """
   Add two half circles to the axes *ax* (or the current axes) with the
   specified facecolors *colors* rotated at *angle* (in degrees).
   """
   if ax is None:
      ax = plt.gca()
   theta1, theta2 = angle, angle + 180
   # w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
   # w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)

   w1 = Wedge(center, radius, theta1, theta2, fc=colors[1], **kwargs)
   w2 = Wedge(center, radius, theta2, theta1, fc=colors[0], **kwargs)

   cr = Circle(center, radius, fc=colors[2], fill=False, **kwargs)
   for wedge in [w1, w2, cr]:
      ax.add_artist(wedge)
   return [w1, w2, cr]


def setup_fig(xlim=(20, -30), ylim=(-20, 20), xlabel='X GSM [Re]', ylabel='Z GSM [Re]'):
   fig = plt.figure(figsize=(15, 10))
   ax = fig.add_subplot(111)
   ax.axvline(0, ls=':', color='k')
   ax.axhline(0, ls=':', color='k')
   ax.set_xlim(xlim)
   ax.set_ylim(ylim)
   ax.set_xlabel(xlabel)
   ax.set_ylabel(ylabel)

   ax.set_aspect('equal')
   w1, w2, cr = dual_half_circle(ax=ax)

   return ax