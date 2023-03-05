import matplotlib.pyplot as plt

from utils.utils import *



indir = r'D:\data\less_sim\b_10_10_1\Results\\'


rad_ = []
for k in range(0,80,10):
    infile = indir + 'thermal_VZ=%d_VA=90'%k
    dbt,ns,nl,nb,geog,proj = read_image_gdal(infile)
    ind = (dbt > 290) * (np.isnan(dbt)!=1)
    rad = planck(10.5,dbt)
    rad = np.average(rad[ind])
    rad_.append(rad)

rad0 = planck(10.5,300)
rad_ = np.asarray(rad_)
emis_1 = rad_/rad0


indir = r'D:\data\less_sim\b_10_10_3\Results\\'


rad_ = []
for k in range(0,80,10):
    infile = indir + 'thermal_VZ=%d_VA=90'%k
    dbt,ns,nl,nb,geog,proj = read_image_gdal(infile)
    ind = (dbt > 290) * (np.isnan(dbt)!=1)
    rad = planck(10.5,dbt)
    rad = np.average(rad[ind])
    rad_.append(rad)

rad0 = planck(10.5,300)
rad_ = np.asarray(rad_)
emis_2 = rad_/rad0


vza_ = np.arange(0,80,10)
plt.plot(vza_,emis_1)
plt.plot(vza_,emis_2)
plt.legend(['hom','het'])
plt.show()
