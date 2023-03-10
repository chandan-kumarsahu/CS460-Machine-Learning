import os
import shutil
import re
import numpy as np
from tqdm import tqdm

# Start with an empty folder named file


ms=[0.8, 1.0, 1.2]
star_mass_files = ['ms'+str(i) for i in ms]

a=[0.1, 0.5, 1.0]
a_files = ['a'+str(i) for i in a]

mp=[(0.5*i) for i in range(1,9)]
#mass_planet_files = ['mp_'+str(i) for i in mp]
r=[(((3*i*5.9722e24)/(4*np.pi*4000))**(1/3))/6.371e6 for i in mp]
rp=[]
for i in range(len(r)):
    rp.append(round(r[i]*0.95, 3))
    rp.append(round(r[i], 3))
    rp.append(round(r[i]*1.05, 3))
#radius_planet_files = ['rp_'+str(0.5*i) for i in rp]
mass_radius_planet_files = ['mp'+str(mp[int(i/3)])+'_rp'+str(rp[i]) for i in range(len(rp))]

wm=[0.5, 1.0, 2.0]
water_mass_files = ['wm'+str(i) for i in wm]

print(len(star_mass_files), len(a_files), len(mass_radius_planet_files), len(water_mass_files))


path='/home/ws1/ML/DATASET3/'

count=0
for i in range(len(star_mass_files)):
	for j in range(len(a_files)):
		for k in range(len(mass_radius_planet_files)):
			for l in range(len(water_mass_files)):
				count=count+1
				name=star_mass_files[i]+'_'+a_files[j]+'_'+mass_radius_planet_files[k]+'_'+water_mass_files[l]
				os.system('mkdir %s' % path+name)
				os.system('cp /home/ws1/ML/Sample_MagmOc_Trappist1g/g.in %s' % path+name+'/'+'planet.in')
				os.system('cp /home/ws1/ML/Sample_MagmOc_Trappist1g/Trappist1.in %s' % path+name+'/'+'star.in')
				os.system('cp /home/ws1/ML/Sample_MagmOc_Trappist1g/vpl.in %s' % path+name+'/'+'vpl.in')

				# open file, read lines
				file1 = open(path+name+'/planet.in', 'r+')
				lines = file1.readlines()
				# split lines into words, change the parameters and merge them back into lines
				lines[1] = re.split(' |\t', lines[1])
				lines[1][3] = 'planet'
				lines[1] = '\t'.join(lines[1])
				lines[6] = re.split(' |\t', lines[6])
				lines[6][3] = str(-mp[int(k/len(rp))])
				lines[6] = '\t'.join(lines[6])
				lines[7] = re.split(' |\t', lines[7])
				lines[7][3] = str(-rp[int(k/len(mp))])
				lines[7] = '\t'.join(lines[7])
				lines[12] = re.split(' |\t', lines[12])
				lines[12][2] = str(-wm[l])
				lines[12] = '\t'.join(lines[12])
				lines[63] = re.split(' |\t', lines[63])
				lines[63][3] = 'star'
				lines[63] = '\t'.join(lines[63])
				lines[68] = re.split(' |\t', lines[68])
				lines[68][3] = str(-a[j])
				lines[68] = '\t'.join(lines[68])
				# write and save into the same file
				file2 = open(path+name+'/planet.in', 'w+')
				file2.writelines(lines)
				file2.close()

				# open file, read lines
				file1 = open(path+name+'/star.in', 'r+')
				lines = file1.readlines()
				# split lines into words, change the parameters and merge them back into lines
				lines[1] = re.split(' |\t', lines[1])
				lines[1][2] = 'star'
				lines[1] = '\t'.join(lines[1])
				lines[8] = re.split(' |\t', lines[8])
				lines[8][2] = str(ms[i])
				lines[8] = '\t'.join(lines[8])
				lines[21] = re.split(' |\t', lines[21])
				lines[21][2] = 'planet'
				lines[21] = '\t'.join(lines[21])
				# write and save into the same file
				file2 = open(path+name+'/star.in', 'w+')
				file2.writelines(lines)
				file2.close()

				# open file, read lines
				file1 = open(path+name+'/vpl.in', 'r+')
				lines = file1.readlines()
				# split lines into words, change the parameters and merge them back into lines
				lines[0] = re.split(' |\t', lines[0])
				lines[0][2] = 'star'
				lines[0] = '\t'.join(lines[0])
				lines[5] = re.split(' |\t', lines[5])
				lines[5][2] = 'star.in'
				lines[5] = '\t'.join(lines[5])
				lines[5] = re.split(' |\t', lines[5])
				lines[5][3] = 'planet.in'
				lines[5] = '\t'.join(lines[5])
				# write and save into the same file
				file2 = open(path+name+'/vpl.in', 'w+')
				file2.writelines(lines)
				file2.close()

print("%d dataset created successfully." %count)


# run the files
dir_path=r'/home/ws1/ML/DATASET3/'
# Iterate directory
for file_path in tqdm(sorted(os.listdir(dir_path))):
    os.chdir(dir_path+file_path)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('#######################  '+file_path+'  ###############################')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    os.system('vplanet vpl.in')
    print('\n\n')




