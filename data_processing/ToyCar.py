import  pandas as pd
#read the csv file
data = pd.read_csv(r'C:\PythonCode\website_toycar.csv')   
print (data)
import numpy as np
#define data in numpy to be able to work with
A= np.array(data)  
import matplotlib.pyplot as plt
import pywt
import scipy.fft as sc
import scipy.signal as scs

import warnings
warnings.filterwarnings("ignore")

import sklearn.svm as sk

#function for Plotting one curve, a=0 or 1 sets label
def plot1ac(t, x, name, a):
    if a==0:
        plt.plot(t, x, label='before denoising')
    else :
        plt.plot(t, x, label='after denoising')
    plt.xlabel("time (millisecond)")
    plt.ylabel("Acceleration") 
    plt.title(name)
    plt.legend() 
    return plt.show()

#function for Plotting one curve without given time scale, a=0 or 1 sets label,
def plot1ac2(x, name, a):
    if a==0:
        plt.plot(x, label='before denoising')
    else :
        plt.plot(x, label='after denoising')
    plt.xlabel("time (millisecond)")
    plt.ylabel("Acceleration") 
    plt.title(name)
    plt.legend() 
    return plt.show()

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

#function for denoising
def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    # aaa=pywt.waverec(coeff, wavelet, mode='per')
    # plt.plot(aaa, label='after denoising')
    # plt.legend() 
    # plt.show()
    return pywt.waverec(coeff, wavelet, mode='per')

#function for fourier transform, with negative amounts
def fftfunction1(signal, duration, N, name):
    yf= sc.fft(signal)
    xf= sc.fftfreq(N, N//duration)
    plt.plot(xf, np.abs(yf))
    plt.title(name)
    return plt.show()

#function for fourier transform, without negative amounts 
def fftfunction(signal, duration, N, name):
    y0f= sc.fft(signal)
    yf= 2.0/N *np.abs(y0f[0:N//2])
    xf= sc.fftfreq(N, N//duration)[:N//2]  #should be duration//n but no answer!because time/1000!
    #plt.plot(xf, yf)
    #plt.title(name)
    #plt.show()
    return xf,yf

def alltop5points(dx,dy,dz,t):
    xx, yx=fftfunction(dx, t[len(t)-1], len(t), 'ac_x')
    xy, yy=fftfunction(dy, t[len(t)-1], len(t), 'ac_y')
    xz, yz=fftfunction(dz, t[len(t)-1], len(t), 'ac_z')
    a1=top5points(xx,yx)
    a2=top5points(xy,yy)
    a3=top5points(xz,yz)
    A=np.concatenate((a1[:,1], a2[:,1], a3[:,1],a1[:,0], a2[:,0], a3[:,0]))
    #[:,1]=amplitude in top points,[:,2]=frequency in top points
    return A

#finction for finding frequency and amplitude of top 5 points
def top5points(xf,yf):    
    #finding index of local maximums
    index= np.array(scs.argrelextrema(yf, np.greater))
    index= np.transpose(index)
    localmax=np.zeros((len(index),2))
    for i in range(0,len(index)):
        localmax[i,0]=xf[index[i]]
        localmax[i,1]=yf[index[i]]    
    #sorting local maximums
    sortlocalmax = localmax[localmax[:,1].argsort()]
    top6localmax=np.delete(sortlocalmax, np.s_[0:len(index)-6], axis = 0)
    #remonve zero frequency
    top5localmax=np.delete(top6localmax, 5, axis = 0)
    return top5localmax

#function for finding energy of signals in 2 ways: with or without zero frequency
def allenergy(dx,dy,dz):
    fx, Px=scs.welch(dx)
    fy, Py=scs.welch(dy)
    fz, Pz=scs.welch(dz)
    # plt.plot(f,Px)
    # plt.show()
    # # plot with y-axis in log scaling
    # plt.semilogy(f, Pxx_den)
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    # plt.show()
    Emean1=[np.mean(Px), np.mean(Py), np.mean(Pz)] #Energy with zero frequency   
    Pxx=np.delete(Px, [0,1,2,3,4,5,6,7,8])
    Pyy=np.delete(Py, [0,1,2,3,4,5,6,7,8])
    Pzz=np.delete(Pz, [0,1,2,3,4,5,6,7,8])
    Emean2=[np.mean(Pxx), np.mean(Pyy), np.mean(Pzz)] #Energy without zero frequency
    return Emean1 #output is Energy with zero frequency

#function for finding energy of signals in 2 ways: with or without zero frequency
def allenergy2(dx,dy,dz):
    fx, Px=scs.welch(dx)
    fy, Py=scs.welch(dy)
    fz, Pz=scs.welch(dz)
    # plt.plot(f,Px)
    # plt.show()
    # # plot with y-axis in log scaling
    # plt.semilogy(f, Pxx_den)
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    # plt.show()
    Emean1=[np.mean(Px)/2, np.mean(Py)/2, np.mean(Pz)/2] #Energy with zero frequency   
    Pxx=np.delete(Px, [0,1,2,3,4,5,6,7,8])
    Pyy=np.delete(Py, [0,1,2,3,4,5,6,7,8])
    Pzz=np.delete(Pz, [0,1,2,3,4,5,6,7,8])
    Emean2=[np.mean(Pxx), np.mean(Pyy), np.mean(Pzz)] #Energy without zero frequency
    return Emean1 #output is Energy with zero frequency

def correlation(dx,dy,dz):
    coxy=np.corrcoef(dx, dy)
    coxz=np.corrcoef(dx, dz)
    coyz=np.corrcoef(dy, dz)
    allcorr=[coxy[0,1], coxz[0,1], coyz[0,1]] 
    return allcorr

#function for short time fourier transform
def stftfunction(signal, name):
    f, t, Zxx = scs.stft(signal,nperseg=8) #64,128,256 v#8
    plt.pcolormesh(t, f/0.5 , np.abs(Zxx))
    plt.title(name)
    plt.ylabel('Normalized Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return f, t, Zxx

#function for jolt Recognition in x axis
def jolt(signalx, name):
    f, t, Zxx= stftfunction(signalx, name)    
    stftmean= np.mean(np.abs(Zxx), axis=0)
    stftmean_normalized= stftmean/np.max(stftmean)
    plt.plot(stftmean_normalized)
    plt.title(name)
    plt.show()
    j=np.std(stftmean_normalized)
    return j

#function for jerk Recognition
def jerk(signalx, name):
    f, t, Zxx= stftfunction(signalx, name)
    Zxx_new=np.abs(Zxx)
    Zxx_new[Zxx_new < 10]= 0
    stftmean= np.mean(np.nonzero(Zxx_new), axis=0)
    stftmean_normalized= stftmean/np.max(stftmean)
    plt.plot(stftmean_normalized)
    plt.title(name)
    plt.show()
    j=np.std(stftmean_normalized)
    return j


print('label 89')
A89 = A[A[:, 7] == 89, :]
#extract all rows with the 8th column 89 
#print (A89)
bias89=A89[0,1]
#bias in  time scale that should be subtracted  
for i in range(0,len(A89)):
    A89[i,1]=A89[i,1]-bias89   
#print (A89)
x89time= A89[:,1]/1000
y89ac=[A89[:,2], A89[:,3], A89[:,4]] #ac_x, ac_y, ac_z
y89encoder= [A89[:,5],A89[:,6]]  #encoder1, encoder2
plot1ac(x89time, y89ac[0],89 ,0) #see the signal before denoising
dy89ac_x=wavelet_denoising(y89ac[0])
dy89ac_y=wavelet_denoising(y89ac[1])
dy89ac_z=wavelet_denoising(y89ac[2])
F89=alltop5points(dy89ac_x, dy89ac_y, dy89ac_z, x89time) #feature vector=FV
F89=np.concatenate((F89,allenergy(dy89ac_x, dy89ac_y, dy89ac_z))) #add energy to FV
F89=np.concatenate((F89,correlation(dy89ac_x, dy89ac_y, dy89ac_z))) #add correlation coefficients to FV
F89=np.append(F89,[jolt(dy89ac_x, '89')])#add xjolt to FV
F89=np.append(F89,[6])
F89=np.append(F89,[jolt(dy89ac_y, '89')])#add yjolt to FV
F89=np.append(F89,[4])
F89=np.append(F89,[x89time[len(x89time)-1]]) #add time feature
#Zxxnew=np.abs(Zxx)
# plot1ac(x89time, y89encoder[0], 89 ,0)
# plot1ac(x89time, y89encoder[1], 89 ,0)
y89encoder=np.array(y89encoder)
newen=np.diff(y89encoder)
newen=np.where(newen>10, 0, newen)
# plt.plot(newen[0])
# plt.ylabel("encoder") 
# plt.legend() 
# plt.show()
F89=np.append(F89,[np.sum(newen)]) #add total number of turns
F89=np.append(F89,[len(newen[newen > 0])]) #add number of chnages in encoder
#F89=np.append(F89,[np.sum(newen)/x89time[len(x89time)-1]]) #add total number of turns in time
F89=np.append(F89,[len(newen[newen > 0])/x89time[len(x89time)-1]]) #add number of chnages in time
F89=np.append(F89,[0,89]) #label normal
F89=np.append(F89,[jolt(dy89ac_z, '89z')])#add yjolt to FV
F89=np.append(F89,[4])

print('label 94')
A94 = A[A[:, 7] == 94, :]
bias94=A94[0,1]
for i in range(0,len(A94)):
      A94[i,1]=A94[i,1]- bias94    
#print (A94)
x94time= A94[:,1]/1000
y94ac=[A94[:,2], A94[:,3], A94[:,4]] 
y94encoder= [A94[:,5],A94[:,6]]
# plot1ac(x94time, dy94ac_x,94 , 1)
dy94ac_x=wavelet_denoising(y94ac[0])
dy94ac_y=wavelet_denoising(y94ac[1])
dy94ac_z=wavelet_denoising(y94ac[2])
F94=alltop5points(dy94ac_x, dy89ac_y, dy94ac_z, x94time) #feature vector=FV
F94=np.concatenate((F94,allenergy(dy94ac_x, dy94ac_y, dy94ac_z))) #add energy to FV
F94=np.concatenate((F94,correlation(dy94ac_x, dy94ac_y, dy94ac_z))) #add correlation coefficients to FV
F94=np.append(F94,[jolt(dy94ac_x, '94')])#add xjolt to FV
F94=np.append(F94,[9]) #8
F94=np.append(F94,[jolt(dy94ac_y, '94')])#add yjolt to FV
F94=np.append(F94,[6])
F94=np.append(F94,[x94time[len(x94time)-1]])
y94encoder=np.array(y94encoder)
newen=np.diff(y94encoder)
newen=np.where(newen>10, 0, newen)
F94=np.append(F94,[np.sum(newen)]) #add total number of turns
F94=np.append(F94,[len(newen[newen > 0])]) #add number of chnages in encoder
F94=np.append(F94,[len(newen[newen > 0])/x94time[len(x94time)-1]])
F94=np.append(F94,[0,94])
F94=np.append(F94,[jolt(dy94ac_z, '94z')])#add yjolt to FV
F94=np.append(F94,[6])

print('label 95')
A95 = A[A[:, 7] == 95, :] 
ChangeLocation=0
#find the location of change in the timescale
for i in range(0,len(A95)):
    if A95[i,1] == 1594:# 1594= initial value of timer        
        ChangeLocation=i        
#print(ChangeLocation)     
#print(A95[ChangeLocation-1,1])
biastime=A95[ChangeLocation-1,1]
for i in range(ChangeLocation,len(A95)):
      A95[i,1]=A95[i,1]+ biastime
biasEn1=A95[ChangeLocation-1,5]
for i in range(ChangeLocation,len(A95)):
      A95[i,5]=A95[i,5]+ biasEn1
biasEn2=A95[ChangeLocation-1,6]
for i in range(ChangeLocation,len(A95)):
      A95[i,6]=A95[i,6]+ biasEn2 
bias95En1=A95[0,5]
for i in range(0,len(A95)):
      A95[i,5]=A95[i,5]- bias95En1   
bias95En2=A95[0,6]
for i in range(0,len(A95)):
      A95[i,6]=A95[i,6]- bias95En2
bias95=A95[0,1]
for i in range(0,len(A95)):
      A95[i,1]=A95[i,1]- bias95
#print (A95)
x95time= A95[:,1]/1000
y95ac=[A95[:,2], A95[:,3], A95[:,4]] 
y95encoder= [A95[:,5],A95[:,6]]
dy95ac_x=wavelet_denoising(y95ac[0])
dy95ac_y=wavelet_denoising(y95ac[1])
dy95ac_z=wavelet_denoising(y95ac[2])
F95=alltop5points(dy95ac_x, dy89ac_y, dy95ac_z, x95time) 
F95=np.concatenate((F95,allenergy(dy95ac_x, dy95ac_y, dy95ac_z))) 
F95=np.concatenate((F95,correlation(dy95ac_x, dy95ac_y, dy95ac_z)))
F95=np.append(F95,[jolt(dy95ac_x, '95')])
F95=np.append(F95,[4])
F95=np.append(F95,[jolt(dy95ac_y, '95')])
F95=np.append(F95,[6])
F95=np.append(F95,[x95time[len(x95time)-1]])
y95encoder=np.array(y95encoder)
newen=np.diff(y95encoder)
newen=np.where(newen>10, 0, newen)
F95=np.append(F95,[np.sum(newen)]) #add total number of turns
F95=np.append(F95,[len(newen[newen > 0])]) #add number of chnages in encoder
F95=np.append(F95,[len(newen[newen > 0])/x95time[len(x95time)-1]])
F95=np.append(F95,[1,95]) #label autistic 
F95=np.append(F95,[jolt(dy95ac_z, '95z')])
F95=np.append(F95,[6])


print('label 100')
A100 = A[A[:, 7] == 100, :]
bias100=A100[0,1]
for i in range(0,len(A100)):
      A100[i,1]=A100[i,1]- bias100    
#print (A100)
x100time= A100[:,1]/1000
y100ac=[A100[:,2], A100[:,3], A100[:,4]] 
y100encoder= [A100[:,5], A100[:,6]]
dy100ac_x=wavelet_denoising(y100ac[0])
dy100ac_y=wavelet_denoising(y100ac[1])
dy100ac_z=wavelet_denoising(y100ac[2])
F100=alltop5points(dy100ac_x, dy89ac_y, dy100ac_z, x100time) 
F100=np.concatenate((F100,allenergy(dy100ac_x, dy100ac_y, dy100ac_z))) 
F100=np.concatenate((F100,correlation(dy100ac_x, dy100ac_y, dy100ac_z))) 
F100=np.append(F100,[jolt(dy100ac_x, '100')])
F100=np.append(F100,[6])
F100=np.append(F100,[jolt(dy100ac_y, '100')])
F100=np.append(F100,[8]) #6
F100=np.append(F100,[x100time[len(x100time)-1]])
y100encoder=np.array(y100encoder)
newen=np.diff(y100encoder)
newen=np.where(newen>10, 0, newen)
F100=np.append(F100,[np.sum(newen)]) #add total number of turns
F100=np.append(F100,[len(newen[newen > 0])]) #add number of chnages in encoder
F100=np.append(F100,[len(newen[newen > 0])/x100time[len(x100time)-1]])
F100=np.append(F100,[1,100])
F100=np.append(F100,[jolt(dy100ac_z, '100z')])
F100=np.append(F100,[8]) #6


print('label 102')
A102 = A[A[:, 7] == 102, :] 
bias102=A102[0,1]
for i in range(0,len(A102)):
      A102[i,1]=A102[i,1]- bias102    
#print (A102)
x102time= A102[:,1]/1000
y102ac=[A102[:,2], A102[:,3], A102[:,4]] 
y102encoder= [A102[:,5], A102[:,6]]
dy102ac_x=wavelet_denoising(y102ac[0])
dy102ac_y=wavelet_denoising(y102ac[1])
dy102ac_z=wavelet_denoising(y102ac[2])
F102=alltop5points(dy102ac_x, dy89ac_y, dy102ac_z, x102time) 
F102=np.concatenate((F102,allenergy(dy102ac_x, dy102ac_y, dy102ac_z))) 
F102=np.concatenate((F102,correlation(dy102ac_x, dy102ac_y, dy102ac_z)))  
F102=np.append(F102,[jolt(dy102ac_x, '102')])
F102=np.append(F102,[4])
F102=np.append(F102,[jolt(dy102ac_y, '102')])
F102=np.append(F102,[2])
F102=np.append(F102,[x102time[len(x102time)-1]])
y102encoder=np.array(y102encoder)
newen=np.diff(y102encoder)
newen=np.where(newen>10, 0, newen)
F102=np.append(F102,[np.sum(newen)]) #add total number of turns
F102=np.append(F102,[len(newen[newen > 0])]) #add number of chnages in encoder
F102=np.append(F102,[len(newen[newen > 0])/x102time[len(x102time)-1]])
F102=np.append(F102,[0,102])
F102=np.append(F102,[jolt(dy102ac_z, '102z')])
F102=np.append(F102,[2])


print('label 106')
A106 = A[A[:, 7] == 106, :] 
bias106=A106[0,1]
for i in range(0,len(A106)):
      A106[i,1]=A106[i,1]- bias106    
#print (A106)
x106time= A106[:,1]/1000
y106ac=[A106[:,2], A106[:,3], A106[:,4]] 
y106encoder= [A106[:,5], A106[:,6]]
dy106ac_x=wavelet_denoising(y106ac[0])
dy106ac_y=wavelet_denoising(y106ac[1])
dy106ac_z=wavelet_denoising(y106ac[2])
F106=alltop5points(dy106ac_x, dy89ac_y, dy106ac_z, x106time) 
F106=np.concatenate((F106,allenergy(dy106ac_x, dy106ac_y, dy106ac_z))) 
F106=np.concatenate((F106,correlation(dy106ac_x, dy106ac_y, dy106ac_z))) 
F106=np.append(F106,[jolt(dy106ac_x, '106')])
F106=np.append(F106,[1]) #2
F106=np.append(F106,[jolt(dy106ac_y, '106')])
F106=np.append(F106,[1]) #2
F106=np.append(F106,[x106time[len(x106time)-1]])
y106encoder=np.array(y106encoder)
newen=np.diff(y106encoder)
newen=np.where(newen>10, 0, newen)
F106=np.append(F106,[np.sum(newen)]) #add total number of turns
F106=np.append(F106,[len(newen[newen > 0])]) #add number of chnages in encoder
F106=np.append(F106,[len(newen[newen > 0])/x106time[len(x106time)-1]])
F106=np.append(F106,[1,106]) 
F106=np.append(F106,[jolt(dy106ac_z, '106z')])
F106=np.append(F106,[1]) #2


print('label 108')
A108 = A[A[:, 7] == 108, :]  
bias108=A108[0,1]
for i in range(0,len(A108)):
      A108[i,1]=A108[i,1]- bias108    
#print (A108)
x108time= A108[:,1]/1000
y108ac=[A108[:,2], A108[:,3], A108[:,4]] 
y108encoder= [A108[:,5], A108[:,6]]
dy108ac_x=wavelet_denoising(y108ac[0])
dy108ac_y=wavelet_denoising(y108ac[1])
dy108ac_z=wavelet_denoising(y108ac[2])
F108=alltop5points(dy108ac_x, dy89ac_y, dy108ac_z, x108time) 
F108=np.concatenate((F108,allenergy(dy108ac_x, dy108ac_y, dy108ac_z)))
F108=np.concatenate((F108,correlation(dy108ac_x, dy108ac_y, dy108ac_z))) 
F108=np.append(F108,[jolt(dy108ac_x, '108')])
F108=np.append(F108,[2])
F108=np.append(F108,[jolt(dy108ac_y, '108')])
F108=np.append(F108,[4])
F108=np.append(F108,[x108time[len(x108time)-1]])
y108encoder=np.array(y108encoder)
newen=np.diff(y108encoder)
newen=np.where(newen>10, 0, newen)
F108=np.append(F108,[np.sum(newen)]) #add total number of turns
F108=np.append(F108,[len(newen[newen > 0])]) #add number of chnages in encoder
F108=np.append(F108,[len(newen[newen > 0])/x108time[len(x108time)-1]])
F108=np.append(F108,[0,108]) #label NA
F108=np.append(F108,[jolt(dy108ac_z, '108z')])
F108=np.append(F108,[4])


print('label 112')
A112 = A[A[:, 7] == 112, :]  
bias112=A112[0,1]
for i in range(0,len(A112)):
      A112[i,1]=A112[i,1]- bias112    
#print (A112)
x112time= A112[:,1]/1000
y112ac=[A112[:,2], A112[:,3], A112[:,4]] 
y112encoder= [A112[:,5], A112[:,6]] 
dy112ac_x=wavelet_denoising(y112ac[0])
dy112ac_y=wavelet_denoising(y112ac[1])
dy112ac_z=wavelet_denoising(y112ac[2])
F112=alltop5points(dy112ac_x, dy89ac_y, dy112ac_z, x112time) 
F112=np.concatenate((F112,allenergy(dy112ac_x, dy112ac_y, dy112ac_z))) 
F112=np.concatenate((F112,correlation(dy112ac_x, dy112ac_y, dy112ac_z))) 
F112=np.append(F112,[jolt(dy112ac_x, '112')])
F112=np.append(F112,[2]) #1
F112=np.append(F112,[jolt(dy112ac_y, '112')])
F112=np.append(F112,[2]) #3
F112=np.append(F112,[x112time[len(x112time)-1]])
y112encoder=np.array(y112encoder)
newen=np.diff(y112encoder)
newen=np.where(newen>10, 0, newen)
F112=np.append(F112,[np.sum(newen)]) #add total number of turns
F112=np.append(F112,[len(newen[newen > 0])]) #add number of chnages in encoder
F112=np.append(F112,[len(newen[newen > 0])/x112time[len(x112time)-1]])
F112=np.append(F112,[1,112])
F112=np.append(F112,[jolt(dy112ac_z, '112z')])
F112=np.append(F112,[2])

print('label 118')
A118 = A[A[:, 7] == 118, :]  
bias118=A118[0,1]
for i in range(0,len(A118)):
      A118[i,1]=A118[i,1]- bias118    
#print (A118)
x118time= A118[:,1]/1000
y118ac=[A118[:,2], A118[:,3], A118[:,4]] 
y118encoder= [A118[:,5], A118[:,6]] 
dy118ac_x=wavelet_denoising(y118ac[0])
dy118ac_y=wavelet_denoising(y118ac[1])
dy118ac_z=wavelet_denoising(y118ac[2])
F118=alltop5points(dy118ac_x, dy89ac_y, dy118ac_z, x118time) 
F118=np.concatenate((F118,allenergy(dy118ac_x, dy118ac_y, dy118ac_z))) 
F118=np.concatenate((F118,correlation(dy118ac_x, dy118ac_y, dy118ac_z))) 
F118=np.append(F118,[jolt(dy118ac_x, '118')])
F118=np.append(F118,[3])
F118=np.append(F118,[jolt(dy118ac_y, '118')])
F118=np.append(F118,[4]) #5
F118=np.append(F118,[x118time[len(x118time)-1]])
y118encoder=np.array(y118encoder)
newen=np.diff(y118encoder)
newen=np.where(newen>10, 0, newen)
F118=np.append(F118,[np.sum(newen)]) #add total number of turns
F118=np.append(F118,[len(newen[newen > 0])]) #add number of chnages in encoder
F118=np.append(F118,[len(newen[newen > 0])/x118time[len(x118time)-1]])
F118=np.append(F118,[1,118]) 
F118=np.append(F118,[jolt(dy118ac_z, '118z')])
F118=np.append(F118,[4])


print('label 120')
A120 = A[A[:, 7] == 120, :] 
bias120=A120[0,1]
for i in range(0,len(A120)):
      A120[i,1]=A120[i,1]- bias120    
#print (A120)
x120time= A120[:,1]/1000
y120ac=[A120[:,2], A120[:,3], A120[:,4]] 
y120encoder= [A120[:,5], A120[:,6]]
dy120ac_x=wavelet_denoising(y120ac[0])
dy120ac_y=wavelet_denoising(y120ac[1])
dy120ac_z=wavelet_denoising(y120ac[2])
F120=alltop5points(dy120ac_x, dy89ac_y, dy120ac_z, x120time) 
F120=np.concatenate((F120,allenergy(dy120ac_x, dy120ac_y, dy120ac_z))) 
F120=np.concatenate((F120,correlation(dy120ac_x, dy120ac_y, dy120ac_z)))
F120=np.append(F120,[jolt(dy120ac_x, '120')])
F120=np.append(F120,[6])
F120=np.append(F120,[jolt(dy120ac_y, '120')])
F120=np.append(F120,[6])
F120=np.append(F120,[x120time[len(x120time)-1]])
y120encoder=np.array(y120encoder)
newen=np.diff(y120encoder)
newen=np.where(newen>10, 0, newen)
F120=np.append(F120,[np.sum(newen)]) #add total number of turns
F120=np.append(F120,[len(newen[newen > 0])]) #add number of chnages in encoder
F120=np.append(F120,[len(newen[newen > 0])/x120time[len(x120time)-1]])
F120=np.append(F120,[0,120]) #label NA
F120=np.append(F120,[jolt(dy120ac_z, '120z')])
F120=np.append(F120,[6])


print('label 121')
A121 = A[A[:, 7] == 121, :] 
bias121=A121[0,1]
for i in range(0,len(A121)):
      A121[i,1]=A121[i,1]- bias121   
#print (A121)
x121time= A121[:,1]/1000
y121ac=[A121[:,2], A121[:,3], A121[:,4]] 
y121encoder= [A121[:,5], A121[:,6]]
dy121ac_x=wavelet_denoising(y121ac[0])
dy121ac_y=wavelet_denoising(y121ac[1])
dy121ac_z=wavelet_denoising(y121ac[2])
F121=alltop5points(dy121ac_x, dy89ac_y, dy121ac_z, x121time) 
F121=np.concatenate((F121,allenergy(dy121ac_x, dy121ac_y, dy121ac_z))) 
F121=np.concatenate((F121,correlation(dy121ac_x, dy121ac_y, dy121ac_z))) 
F121=np.append(F121,[jolt(dy121ac_x, '121')])
F121=np.append(F121,[3])
F121=np.append(F121,[jolt(dy121ac_y, '121')])
F121=np.append(F121,[5])
F121=np.append(F121,[x121time[len(x121time)-1]])
y121encoder=np.array(y121encoder)
newen=np.diff(y121encoder)
newen=np.where(newen>10, 0, newen)
F121=np.append(F121,[np.sum(newen)]) #add total number of turns
F121=np.append(F121,[len(newen[newen > 0])]) #add number of chnages in encoder
F121=np.append(F121,[len(newen[newen > 0])/x121time[len(x121time)-1]])
F121=np.append(F121,[0,121]) #label NA
F121=np.append(F121,[jolt(dy121ac_z, '121')])


print('label 124')
A124 = A[A[:, 7] == 124, :]  
bias124=A124[0,1]
for i in range(0,len(A124)):
      A124[i,1]=A124[i,1]- bias124   
#print (A124)
x124time= A124[:,1]/1000
y124ac=[A124[:,2], A124[:,3], A124[:,4]] 
y124encoder= [A124[:,5], A124[:,6]]
dy124ac_x=wavelet_denoising(y124ac[0])
dy124ac_y=wavelet_denoising(y124ac[1])
dy124ac_z=wavelet_denoising(y124ac[2])
F124=alltop5points(dy124ac_x, dy89ac_y, dy124ac_z, x124time) 
F124=np.concatenate((F124,allenergy(dy124ac_x, dy124ac_y, dy124ac_z))) 
F124=np.concatenate((F124,correlation(dy124ac_x, dy124ac_y, dy124ac_z))) 
F124=np.append(F124,[jolt(dy124ac_x, '124')])
F124=np.append(F124,[4]) #3
F124=np.append(F124,[jolt(dy124ac_y, '124')])
F124=np.append(F124,[4]) #5
F124=np.append(F124,[x124time[len(x124time)-1]])
y124encoder=np.array(y124encoder)
newen=np.diff(y124encoder)
newen=np.where(newen>10, 0, newen)
F124=np.append(F124,[np.sum(newen)]) #add total number of turns
F124=np.append(F124,[len(newen[newen > 0])]) #add number of chnages in encoder
F124=np.append(F124,[len(newen[newen > 0])/x124time[len(x124time)-1]])
F124=np.append(F124,[0,124]) #label NA
F124=np.append(F124,[jolt(dy124ac_z, '124')])


print('label 201')
A201 = pd.read_csv(r'C:\PythonCode\A_amirali.csv')   
#print (A201)
A201=np.array(A201)
A201=np.delete(A201, [1394], 0)
A201 = np.vstack(A201[:, :]).astype(np.float)
#print (A201)
x201time= A201[:,0]
y201ac=[A201[:,1], A201[:,2], A201[:,3]] #ac_x, ac_y, ac_z
y201encoder= [A201[:,4],A201[:,5]]  #encoder1, encoder2
plot1ac(x201time, y201ac[0],89 ,0) #see the signal before denoising
dy201ac_x=wavelet_denoising(y201ac[0])
dy201ac_y=wavelet_denoising(y201ac[1])
dy201ac_z=wavelet_denoising(y201ac[2])
#plot1ac(x201time, y201ac[0], 201, 0)
#plot1ac(x201time, dy201ac_x, 201, 1)
F201=alltop5points(dy201ac_x, dy201ac_y, dy201ac_z, x201time) #feature vector=FV
F201=np.concatenate((F201,allenergy(dy201ac_x, dy201ac_y, dy201ac_z))) #add energy to FV
F201=np.concatenate((F201,correlation(dy201ac_x, dy201ac_y, dy201ac_z))) #add correlation coefficients to FV
F201=np.append(F201,[jolt(dy201ac_x, '201')])#add jolt to FV
F201=np.append(F201,[3])
F201=np.append(F201,[jolt(dy201ac_y, '201')])#add jolt to FV
F201=np.append(F201,[2]) #3
F201=np.append(F201,[x201time[len(x201time)-1]])
y201encoder=np.array(y201encoder)
newen=np.diff(y201encoder)
newen=np.where(newen>10, 0, newen)
F201=np.append(F201,[np.sum(newen)]) #add total number of turns
F201=np.append(F201,[len(newen[newen > 0])]) #add number of chnages in encoder
F201=np.append(F201,[len(newen[newen > 0])/x201time[len(x201time)-1]])
F201=np.append(F201,[1,201]) #label autistic
F201=np.append(F201,[jolt(dy201ac_z, '201')])#add jolt to FV


print('label 202')
A202 = pd.read_csv(r'C:\PythonCode\A_artin-1.csv')   
A202=np.array(A202)
A202=np.delete(A202, [3622], 0)
A202 = np.vstack(A202[:, :]).astype(np.float)
x202time= A202[:,0]
y202ac=[A202[:,1], A202[:,2], A202[:,3]] #ac_x, ac_y, ac_z
y202encoder= [A202[:,4],A202[:,5]]  #encoder1, encoder2
plot1ac(x202time, y202ac[0],89 ,0) #see the signal before denoising
dy202ac_x=wavelet_denoising(y202ac[0])
dy202ac_y=wavelet_denoising(y202ac[1])
dy202ac_z=wavelet_denoising(y202ac[2])
#plot1ac(x202time, y202ac[0], 202, 0)
#plot1ac(x202time, dy202ac_x, 202, 1)
F202=alltop5points(dy202ac_x, dy202ac_y, dy202ac_z, x202time) #feature vector=FV
F202=np.concatenate((F202,allenergy(dy202ac_x, dy202ac_y, dy202ac_z))) #add energy to FV
F202=np.concatenate((F202,correlation(dy202ac_x, dy202ac_y, dy202ac_z))) #add correlation coefficients to FV
F202=np.append(F202,[jolt(dy202ac_x, '202')])#add jolt to FV
F202=np.append(F202,[6])
F202=np.append(F202,[jolt(dy202ac_y, '202')])#add jolt to FV
F202=np.append(F202,[10])
F202=np.append(F202,[x202time[len(x202time)-1]])
y202encoder=np.array(y202encoder)
newen=np.diff(y202encoder)
newen=np.where(newen>10, 0, newen)
F202=np.append(F202,[np.sum(newen)]) #add total number of turns
F202=np.append(F202,[len(newen[newen > 0])]) #add number of chnages in encoder
F202=np.append(F202,[len(newen[newen > 0])/x202time[len(x202time)-1]])
F202=np.append(F202,[1,202]) #label autistic
F202=np.append(F202,[jolt(dy202ac_z, '202')])#add jolt to FV


print('label 203')
A203 = pd.read_csv(r'C:\PythonCode\A_meshkat  alekazem.csv')   
A203=np.array(A203)
A203=np.delete(A203, [2179], 0)
A203 = np.vstack(A203[:, :]).astype(np.float)
x203time= A203[:,0]
y203ac=[A203[:,1], A203[:,2], A203[:,3]] #ac_x, ac_y, ac_z
y203encoder= [A203[:,4],A203[:,5]]  #encoder1, encoder2
plot1ac(x203time, y203ac[0],89 ,0) #see the signal before denoising
dy203ac_x=wavelet_denoising(y203ac[0])
dy203ac_y=wavelet_denoising(y203ac[1])
dy203ac_z=wavelet_denoising(y203ac[2])
#plot1ac(x203time, y203ac[0], 203, 0)
#plot1ac(x203time, dy203ac_x, 203, 1)
F203=alltop5points(dy203ac_x, dy203ac_y, dy203ac_z, x203time) #feature vector=FV
F203=np.concatenate((F203,allenergy(dy203ac_x, dy203ac_y, dy203ac_z))) #add energy to FV
F203=np.concatenate((F203,correlation(dy203ac_x, dy203ac_y, dy203ac_z))) #add correlation coefficients to FV
F203=np.append(F203,[jolt(dy203ac_x, '203')])#add jolt to FV
F203=np.append(F203,[3]) #2
F203=np.append(F203,[jolt(dy203ac_y, '203')])#add jolt to FV
F203=np.append(F203,[1])
F203=np.append(F203,[x203time[len(x203time)-1]])
y203encoder=np.array(y203encoder)
newen=np.diff(y203encoder)
newen=np.where(newen>10, 0, newen)
F203=np.append(F203,[np.sum(newen)]) #add total number of turns
F203=np.append(F203,[len(newen[newen > 0])]) #add number of chnages in encoder
F203=np.append(F203,[len(newen[newen > 0])/x203time[len(x203time)-1]])
F203=np.append(F203,[1,203]) #label autistic
F203=np.append(F203,[jolt(dy203ac_z, '203')])#add jolt to FV

print('label 204')
A204 = pd.read_csv(r'C:\PythonCode\A_keyvan maleki.csv')   
A204=np.array(A204)
A204=np.delete(A204, [4525], 0)
A204 = np.vstack(A204[:, :]).astype(np.float)
x204time= A204[:,0]
y204ac=[A204[:,1], A204[:,2], A204[:,3]] #ac_x, ac_y, ac_z
y204encoder= [A204[:,4],A204[:,5]]  #encoder1, encoder2
plot1ac(x204time, y204ac[0],89 ,0) #see the signal before denoising
dy204ac_x=wavelet_denoising(y204ac[0])
dy204ac_y=wavelet_denoising(y204ac[1])
dy204ac_z=wavelet_denoising(y204ac[2])
#plot1ac(x204time, y204ac[0], 204, 0)
#plot1ac(x204time, dy204ac_x, 204, 1)
F204=alltop5points(dy204ac_x, dy204ac_y, dy204ac_z, x204time) #feature vector=FV
F204=np.concatenate((F204,allenergy(dy204ac_x, dy204ac_y, dy204ac_z))) #add energy to FV
F204=np.concatenate((F204,correlation(dy204ac_x, dy204ac_y, dy204ac_z))) #add correlation coefficients to FV
F204=np.append(F204,[jolt(dy204ac_x, '204')])#add jolt to FV
F204=np.append(F204,[3])
F204=np.append(F204,[jolt(dy204ac_y, '204')])#add jolt to FV
F204=np.append(F204,[5])
F204=np.append(F204,[x204time[len(x204time)-1]])
y204encoder=np.array(y204encoder)
newen=np.diff(y204encoder)
newen=np.where(newen>10, 0, newen)
F204=np.append(F204,[np.sum(newen)]) #add total number of turns
F204=np.append(F204,[len(newen[newen > 0])]) #add number of chnages in encoder
F204=np.append(F204,[len(newen[newen > 0])/x204time[len(x204time)-1]])
F204=np.append(F204,[1,204]) #label autistic
F204=np.append(F204,[jolt(dy204ac_z, '204')])#add jolt to FV


print('label 205')
A205 = pd.read_csv(r'C:\PythonCode\A_nika ziaee.csv')   
A205=np.array(A205)
A205=np.delete(A205, [5347], 0)
A205 = np.vstack(A205[:, :]).astype(np.float)
x205time= A205[:,0]
y205ac=[A205[:,1], A205[:,2], A205[:,3]] #ac_x, ac_y, ac_z
y205encoder= [A205[:,4],A205[:,5]]  #encoder1, encoder2
plot1ac(x205time, y205ac[0],89 ,0) #see the signal before denoising
dy205ac_x=wavelet_denoising(y205ac[0])
dy205ac_y=wavelet_denoising(y205ac[1])
dy205ac_z=wavelet_denoising(y205ac[2])
#plot1ac(x205time, y205ac[0], 205, 0)
#plot1ac(x205time, dy205ac_x, 205, 1)
F205=alltop5points(dy205ac_x, dy205ac_y, dy205ac_z, x205time) #feature vector=FV
F205=np.concatenate((F205,allenergy(dy205ac_x, dy205ac_y, dy205ac_z))) #add energy to FV
F205=np.concatenate((F205,correlation(dy205ac_x, dy205ac_y, dy205ac_z))) #add correlation coefficients to FV
F205=np.append(F205,[jolt(dy205ac_x, '205')])#add jolt to FV
F205=np.append(F205,[5])
F205=np.append(F205,[jolt(dy205ac_y, '205')])#add jolt to FV
F205=np.append(F205,[7])
F205=np.append(F205,[x205time[len(x205time)-1]])
y205encoder=np.array(y205encoder)
newen=np.diff(y205encoder)
newen=np.where(newen>10, 0, newen)
F205=np.append(F205,[np.sum(newen)]) #add total number of turns
F205=np.append(F205,[len(newen[newen > 0])]) #add number of chnages in encoder
F205=np.append(F205,[len(newen[newen > 0])/x205time[len(x205time)-1]])
F205=np.append(F205,[1,205]) #label autistic
F205=np.append(F205,[jolt(dy205ac_z, '205')])


print('label 206')
A206 = pd.read_csv(r'C:\PythonCode\A_arad.csv')   
A206=np.array(A206)
A206=np.delete(A206, [408], 0)
A206 = np.vstack(A206[:, :]).astype(np.float)
x206time= A206[:,0]
y206ac=[A206[:,1], A206[:,2], A206[:,3]] #ac_x, ac_y, ac_z
y206encoder= [A206[:,4],A206[:,5]]  #encoder1, encoder2
plot1ac(x206time, y206ac[0],89 ,0) #see the signal before denoising
dy206ac_x=wavelet_denoising(y206ac[0])
dy206ac_y=wavelet_denoising(y206ac[1])
dy206ac_z=wavelet_denoising(y206ac[2])
#plot1ac(x206time, y206ac[0], 206, 0)
#plot1ac(x206time, dy206ac_x, 206, 1)
F206=alltop5points(dy206ac_x, dy206ac_y, dy206ac_z, x206time) #feature vector=FV
F206=np.concatenate((F206,allenergy(dy206ac_x, dy206ac_y, dy206ac_z))) #add energy to FV
F206=np.concatenate((F206,correlation(dy206ac_x, dy206ac_y, dy206ac_z))) #add correlation coefficients to FV
F206=np.append(F206,[jolt(dy206ac_x, '206')])#add jolt to FV
F206=np.append(F206,[1])
F206=np.append(F206,[jolt(dy206ac_y, '206')])#add jolt to FV
F206=np.append(F206,[2])
F206=np.append(F206,[x206time[len(x206time)-1]])
y206encoder=np.array(y206encoder)
newen=np.diff(y206encoder)
newen=np.where(newen>10, 0, newen)
F206=np.append(F206,[np.sum(newen)]) #add total number of turns
F206=np.append(F206,[len(newen[newen > 0])]) #add number of chnages in encoder
F206=np.append(F206,[len(newen[newen > 0])/x206time[len(x206time)-1]])
F206=np.append(F206,[1,206]) #label autistic
F206=np.append(F206,[jolt(dy206ac_z, '206')])#add jolt to FV


print('label 207')
A207 = pd.read_csv(r'C:\PythonCode\A_abolfazl.csv')   
A207=np.array(A207)
A207 = np.vstack(A207[:, :]).astype(np.float)
x207time= A207[:,0]
y207ac=[A207[:,1], A207[:,2], A207[:,3]] #ac_x, ac_y, ac_z
y207encoder= [A207[:,4],A207[:,5]]  #encoder1, encoder2
plot1ac(x207time, y207ac[0],89 ,0) #see the signal before denoising
dy207ac_x=wavelet_denoising(y207ac[0])
dy207ac_y=wavelet_denoising(y207ac[1])
dy207ac_z=wavelet_denoising(y207ac[2])
#plot1ac(x207time, y207ac[0], 207, 0)
#plot1ac(x207time, dy207ac_x, 207, 1)
F207=alltop5points(dy207ac_x, dy207ac_y, dy207ac_z, x207time) #feature vector=FV
F207=np.concatenate((F207,allenergy(dy207ac_x, dy207ac_y, dy207ac_z))) #add energy to FV
F207=np.concatenate((F207,correlation(dy207ac_x, dy207ac_y, dy207ac_z))) #add correlation coefficients to FV
F207=np.append(F207,[jolt(dy207ac_x, '207')])#add jolt to FV
F207=np.append(F207,[2])
F207=np.append(F207,[jolt(dy207ac_y, '207')])#add jolt to FV
F207=np.append(F207,[2]) #2
F207=np.append(F207,[x207time[len(x207time)-1]])
y207encoder=np.array(y207encoder)
newen=np.diff(y207encoder)
newen=np.where(newen>10, 0, newen)
F207=np.append(F207,[np.sum(newen)]) #add total number of turns
F207=np.append(F207,[len(newen[newen > 0])]) #add number of chnages in encoder
F207=np.append(F207,[len(newen[newen > 0])/x207time[len(x207time)-1]])
F207=np.append(F207,[1,207]) #label autistic
F207=np.append(F207,[jolt(dy207ac_z, '207')])#add jolt to FV


print('label 208')
A208 = pd.read_csv(r'C:\PythonCode\A_adrian abdolahzadeh.csv')   
A208=np.array(A208)
A208 = np.vstack(A208[:, :]).astype(np.float)
x208time= A208[:,0]
y208ac=[A208[:,1], A208[:,2], A208[:,3]] #ac_x, ac_y, ac_z
y208encoder= [A208[:,4],A208[:,5]]  #encoder1, encoder2
plot1ac(x208time, y208ac[0],89 ,0) #see the signal before denoising
dy208ac_x=wavelet_denoising(y208ac[0])
dy208ac_y=wavelet_denoising(y208ac[1])
dy208ac_z=wavelet_denoising(y208ac[2])
#plot1ac(x208time, y208ac[0], 208, 0)
#plot1ac(x208time, dy208ac_x, 208, 1)
F208=alltop5points(dy208ac_x, dy208ac_y, dy208ac_z, x208time) #feature vector=FV
F208=np.concatenate((F208,allenergy(dy208ac_x, dy208ac_y, dy208ac_z))) #add energy to FV
F208=np.concatenate((F208,correlation(dy208ac_x, dy208ac_y, dy208ac_z))) #add correlation coefficients to FV
F208=np.append(F208,[jolt(dy208ac_x, '208')])#add jolt to FV
F208=np.append(F208,[4]) #5
F208=np.append(F208,[jolt(dy208ac_y, '208')])#add jolt to FV
F208=np.append(F208,[3])
F208=np.append(F208,[x208time[len(x208time)-1]])
y208encoder=np.array(y208encoder)
newen=np.diff(y208encoder)
newen=np.where(newen>10, 0, newen)
F208=np.append(F208,[np.sum(newen)]) #add total number of turns
F208=np.append(F208,[len(newen[newen > 0])]) #add number of chnages in encoder
F208=np.append(F208,[len(newen[newen > 0])/x208time[len(x208time)-1]])
F208=np.append(F208,[1,208]) #label autistic
F208=np.append(F208,[jolt(dy208ac_z, '208')])#add jolt to FV


print('label 209')
A209 = pd.read_csv(r'C:\PythonCode\A_arian hashemi.csv')   
A209=np.array(A209)
A209 = np.vstack(A209[:, :]).astype(np.float)
x209time= A209[:,0]
y209ac=[A209[:,1], A209[:,2], A209[:,3]] #ac_x, ac_y, ac_z
y209encoder= [A209[:,4],A209[:,5]]  #encoder1, encoder2
plot1ac(x209time, y209ac[0],89 ,0) #see the signal before denoising
dy209ac_x=wavelet_denoising(y209ac[0])
dy209ac_y=wavelet_denoising(y209ac[1])
dy209ac_z=wavelet_denoising(y209ac[2])
#plot1ac(x209time, y209ac[0], 209, 0)
#plot1ac(x209time, dy209ac_x, 209, 1)
F209=alltop5points(dy209ac_x, dy209ac_y, dy209ac_z, x209time) #feature vector=FV
F209=np.concatenate((F209,allenergy(dy209ac_x, dy209ac_y, dy209ac_z))) #add energy to FV
F209=np.concatenate((F209,correlation(dy209ac_x, dy209ac_y, dy209ac_z))) #add correlation coefficients to FV
F209=np.append(F209,[jolt(dy209ac_x, '209')])#add jolt to FV
F209=np.append(F209,[4]) #5
F209=np.append(F209,[jolt(dy209ac_y, '209')])#add jolt to FV
F209=np.append(F209,[4]) #4
F209=np.append(F209,[x209time[len(x209time)-1]])
y209encoder=np.array(y209encoder)
newen=np.diff(y209encoder)
newen=np.where(newen>10, 0, newen)
F209=np.append(F209,[np.sum(newen)]) #add total number of turns
F209=np.append(F209,[len(newen[newen > 0])]) #add number of chnages in encoder
F209=np.append(F209,[len(newen[newen > 0])/x209time[len(x209time)-1]])
F209=np.append(F209,[1,209]) #label autistic
F209=np.append(F209,[jolt(dy209ac_z, '209')])#add jolt to FV
F209=np.append(F209,[4])

print('label 210')
A210 = pd.read_csv(r'C:\PythonCode\A_ayleen bidar.csv')   
A210=np.array(A210)
A210=np.delete(A205, [3439], 0)
A210 = np.vstack(A210[:, :]).astype(np.float)
x210time= A210[:,0]
y210ac=[A210[:,1], A210[:,2], A210[:,3]] #ac_x, ac_y, ac_z
y210encoder= [A210[:,4],A210[:,5]]  #encoder1, encoder2
plot1ac(x210time, y210ac[0],89 ,0) #see the signal before denoising
dy210ac_x=wavelet_denoising(y210ac[0])
dy210ac_y=wavelet_denoising(y210ac[1])
dy210ac_z=wavelet_denoising(y210ac[2])
#plot1ac(x210time, y210ac[0], 210, 0)
#plot1ac(x210time, dy210ac_x, 210, 1)
F210=alltop5points(dy210ac_x, dy210ac_y, dy210ac_z, x210time) #feature vector=FV
F210=np.concatenate((F210,allenergy(dy210ac_x, dy210ac_y, dy210ac_z))) #add energy to FV
F210=np.concatenate((F210,correlation(dy210ac_x, dy210ac_y, dy210ac_z))) #add correlation coefficients to FV
F210=np.append(F210,[jolt(dy210ac_x, '210')])#add jolt to FV
F210=np.append(F210,[6])
F210=np.append(F210,[jolt(dy210ac_y, '210')])#add jolt to FV
F210=np.append(F210,[7])
F210=np.append(F210,[x210time[len(x210time)-1]])
y210encoder=np.array(y210encoder)
newen=np.diff(y210encoder)
newen=np.where(newen>10, 0, newen)
F210=np.append(F210,[np.sum(newen)]) #add total number of turns
F210=np.append(F210,[len(newen[newen > 0])]) #add number of chnages in encoder
F210=np.append(F210,[len(newen[newen > 0])/x210time[len(x210time)-1]])
F210=np.append(F210,[1,210]) #label autistic
F210=np.append(F210,[jolt(dy210ac_z, '210')])#add jolt to FV
F210=np.append(F210,[7])


print('label 211')
A211 = pd.read_csv(r'C:\PythonCode\A_ayleen kamani.csv')   
A211=np.array(A211)
A211 = np.vstack(A211[:, :]).astype(np.float)
x211time= A211[:,0]
y211ac=[A211[:,1], A211[:,2], A211[:,3]] #ac_x, ac_y, ac_z
y211encoder= [A211[:,4],A211[:,5]]  #encoder1, encoder2
plot1ac(x211time, y211ac[0],89 ,0) #see the signal before denoising
dy211ac_x=wavelet_denoising(y211ac[0])
dy211ac_y=wavelet_denoising(y211ac[1])
dy211ac_z=wavelet_denoising(y211ac[2])
#plot1ac(x211time, y211ac[0], 211, 0)
#plot1ac(x211time, dy211ac_x, 211, 1)
F211=alltop5points(dy211ac_x, dy211ac_y, dy211ac_z, x211time) #feature vector=FV
F211=np.concatenate((F211,allenergy(dy211ac_x, dy211ac_y, dy211ac_z))) #add energy to FV
F211=np.concatenate((F211,correlation(dy211ac_x, dy211ac_y, dy211ac_z))) #add correlation coefficients to FV
F211=np.append(F211,[jolt(dy211ac_x, '211')])#add jolt to FV
F211=np.append(F211,[6]) #5
F211=np.append(F211,[jolt(dy211ac_y, '211')])#add jolt to FV
F211=np.append(F211,[5])
F211=np.append(F211,[x211time[len(x211time)-1]])
y211encoder=np.array(y211encoder)
newen=np.diff(y211encoder)
newen=np.where(newen>10, 0, newen)
F211=np.append(F211,[np.sum(newen)]) #add total number of turns
F211=np.append(F211,[len(newen[newen > 0])]) #add number of chnages in encoder
F211=np.append(F211,[len(newen[newen > 0])/x211time[len(x211time)-1]])
F211=np.append(F211,[1,211]) #label autistic
F211=np.append(F211,[jolt(dy211ac_z, '211')])#add jolt to FV


print('label 212')
A212 = pd.read_csv(r'C:\PythonCode\A_golsa yadegari.csv')   
A212=np.array(A212)
A212 = np.vstack(A212[:, :]).astype(np.float)
x212time= A212[:,0]
y212ac=[A212[:,1], A212[:,2], A212[:,3]] #ac_x, ac_y, ac_z
y212encoder= [A212[:,4],A212[:,5]]  #encoder1, encoder2
plot1ac(x212time, y212ac[0],89 ,0) #see the signal before denoising
dy212ac_x=wavelet_denoising(y212ac[0])
dy212ac_y=wavelet_denoising(y212ac[1])
dy212ac_z=wavelet_denoising(y212ac[2])
#plot1ac(x212time, y212ac[0], 212, 0)
#plot1ac(x212time, dy212ac_x, 212, 1)
F212=alltop5points(dy212ac_x, dy212ac_y, dy212ac_z, x212time) #feature vector=FV
F212=np.concatenate((F212,allenergy(dy212ac_x, dy212ac_y, dy212ac_z))) #add energy to FV
F212=np.concatenate((F212,correlation(dy212ac_x, dy212ac_y, dy212ac_z))) #add correlation coefficients to FV
F212=np.append(F212,[jolt(dy212ac_x, '212')])#add jolt to FV
F212=np.append(F212,[6]) #5
F212=np.append(F212,[jolt(dy212ac_y, '212')])#add jolt to FV
F212=np.append(F212,[7]) #6
F212=np.append(F212,[x212time[len(x212time)-1]])
y212encoder=np.array(y212encoder)
newen=np.diff(y212encoder)
newen=np.where(newen>10, 0, newen)
F212=np.append(F212,[np.sum(newen)]) #add total number of turns
F212=np.append(F212,[len(newen[newen > 0])]) #add number of chnages in encoder
F212=np.append(F212,[len(newen[newen > 0])/x212time[len(x212time)-1]])
F212=np.append(F212,[1,212]) #label autistic
F212=np.append(F212,[jolt(dy212ac_z, '212')])#add jolt to FV
F212=np.append(F212,[7]) #6


print('label 213')
A213 = pd.read_csv(r'C:\PythonCode\A_kasra bahmani.csv')   
A213=np.array(A213)
A213 = np.vstack(A213[:, :]).astype(np.float)
x213time= A213[:,0]
y213ac=[A213[:,1], A213[:,2], A213[:,3]] #ac_x, ac_y, ac_z
y213encoder= [A213[:,4],A213[:,5]]  #encoder1, encoder2
plot1ac(x213time, y213ac[0],89 ,0) #see the signal before denoising
dy213ac_x=wavelet_denoising(y213ac[0])
dy213ac_y=wavelet_denoising(y213ac[1])
dy213ac_z=wavelet_denoising(y213ac[2])
#plot1ac(x213time, y213ac[0], 213, 0)
#plot1ac(x213time, dy213ac_x, 213, 1)
F213=alltop5points(dy213ac_x, dy213ac_y, dy213ac_z, x213time) #feature vector=FV
F213=np.concatenate((F213,allenergy(dy213ac_x, dy213ac_y, dy213ac_z))) #add energy to FV
F213=np.concatenate((F213,correlation(dy213ac_x, dy213ac_y, dy213ac_z))) #add correlation coefficients to FV
F213=np.append(F213,[jolt(dy213ac_x, '213')])#add jolt to FV
F213=np.append(F213,[3]) #4
F213=np.append(F213,[jolt(dy213ac_y, '213')])#add jolt to FV
F213=np.append(F213,[4])
F213=np.append(F213,[x213time[len(x213time)-1]])
y213encoder=np.array(y213encoder)
newen=np.diff(y213encoder)
newen=np.where(newen>10, 0, newen)
F213=np.append(F213,[np.sum(newen)]) #add total number of turns
F213=np.append(F213,[len(newen[newen > 0])]) #add number of chnages in encoder
F213=np.append(F213,[len(newen[newen > 0])/x213time[len(x213time)-1]])
F213=np.append(F213,[1,213]) #label autistic
F213=np.append(F213,[jolt(dy213ac_z, '213')])#add jolt to FV
F213=np.append(F213,[4])


print('label 214')
A214 = pd.read_csv(r'C:\PythonCode\A_mahdiar.csv')   
A214=np.array(A214)
A214 = np.vstack(A214[:, :]).astype(np.float)
x214time= A214[:,0]
y214ac=[A214[:,1], A214[:,2], A214[:,3]] #ac_x, ac_y, ac_z
y214encoder= [A214[:,4],A214[:,5]]  #encoder1, encoder2
plot1ac(x214time, y214ac[0],89 ,0) #see the signal before denoising
dy214ac_x=wavelet_denoising(y214ac[0])
dy214ac_y=wavelet_denoising(y214ac[1])
dy214ac_z=wavelet_denoising(y214ac[2])
#plot1ac(x214time, y214ac[0], 214, 0)
#plot1ac(x214time, dy214ac_x, 214, 1)
F214=alltop5points(dy214ac_x, dy214ac_y, dy214ac_z, x214time) #feature vector=FV
F214=np.concatenate((F214,allenergy(dy214ac_x, dy214ac_y, dy214ac_z))) #add energy to FV
F214=np.concatenate((F214,correlation(dy214ac_x, dy214ac_y, dy214ac_z))) #add correlation coefficients to FV
F214=np.append(F214,[jolt(dy214ac_x, '214')])#add jolt to FV
F214=np.append(F214,[1]) #2
F214=np.append(F214,[jolt(dy214ac_y, '214')])#add jolt to FV
F214=np.append(F214,[3])
F214=np.append(F214,[x214time[len(x214time)-1]])
y214encoder=np.array(y214encoder)
newen=np.diff(y214encoder)
newen=np.where(newen>10, 0, newen)
F214=np.append(F214,[np.sum(newen)]) #add total number of turns
F214=np.append(F214,[len(newen[newen > 0])]) #add number of chnages in encoder
F214=np.append(F214,[len(newen[newen > 0])/x214time[len(x214time)-1]])
F214=np.append(F214,[1,214]) #label autistic
F214=np.append(F214,[jolt(dy214ac_z, '214')])#add jolt to FV
F214=np.append(F214,[3])


print('label 215')
A215 = pd.read_csv(r'C:\PythonCode\A_matin moghtaderian.csv')   
#print (A215)
A215=np.array(A215)
A215 = np.vstack(A215[:, :]).astype(np.float)
#print (A215)
x215time= A215[:,0]
y215ac=[A215[:,1], A215[:,2], A215[:,3]] #ac_x, ac_y, ac_z
y215encoder= [A215[:,4],A215[:,5]]  #encoder1, encoder2
plot1ac(x215time, y215ac[0],89 ,0) #see the signal before denoising
dy215ac_x=wavelet_denoising(y215ac[0])
dy215ac_y=wavelet_denoising(y215ac[1])
dy215ac_z=wavelet_denoising(y215ac[2])
#plot1ac(x215time, y215ac[0], 215, 0)
#plot1ac(x215time, dy215ac_x, 215, 1)
F215=alltop5points(dy215ac_x, dy215ac_y, dy215ac_z, x215time) #feature vector=FV
F215=np.concatenate((F215,allenergy(dy215ac_x, dy215ac_y, dy215ac_z))) #add energy to FV
F215=np.concatenate((F215,correlation(dy215ac_x, dy215ac_y, dy215ac_z))) #add correlation coefficients to FV
F215=np.append(F215,[jolt(dy215ac_x, '215')])#add jolt to FV
F215=np.append(F215,[3]) #4
F215=np.append(F215,[jolt(dy215ac_y, '215')])#add jolt to FV
F215=np.append(F215,[6])
F215=np.append(F215,[x215time[len(x215time)-1]])
y215encoder=np.array(y215encoder)
newen=np.diff(y215encoder)
newen=np.where(newen>10, 0, newen)
F215=np.append(F215,[np.sum(newen)]) #add total number of turns
F215=np.append(F215,[len(newen[newen > 0])]) #add number of chnages in encoder
F215=np.append(F215,[len(newen[newen > 0])/x215time[len(x215time)-1]])
F215=np.append(F215,[1,215]) #label autistic
F215=np.append(F215,[jolt(dy215ac_z, '215')])#add jolt to FV
F215=np.append(F215,[6])


print('label 216')
A216 = pd.read_csv(r'C:\PythonCode\A_narjes dow.csv')   
#print (A216)
A216=np.array(A216)
A216 = np.vstack(A216[:, :]).astype(np.float)
#print (A216)
x216time= A216[:,0]
y216ac=[A216[:,1], A216[:,2], A216[:,3]] #ac_x, ac_y, ac_z
y216encoder= [A216[:,4],A216[:,5]]  #encoder1, encoder2
plot1ac(x216time, y216ac[0],89 ,0) #see the signal before denoising
dy216ac_x=wavelet_denoising(y216ac[0])
dy216ac_y=wavelet_denoising(y216ac[1])
dy216ac_z=wavelet_denoising(y216ac[2])
#plot1ac(x216time, y216ac[0], 216, 0)
#plot1ac(x216time, dy216ac_x, 216, 1)
F216=alltop5points(dy216ac_x, dy216ac_y, dy216ac_z, x216time) #feature vector=FV
F216=np.concatenate((F216,allenergy(dy216ac_x, dy216ac_y, dy216ac_z))) #add energy to FV
F216=np.concatenate((F216,correlation(dy216ac_x, dy216ac_y, dy216ac_z))) #add correlation coefficients to FV
F216=np.append(F216,[jolt(dy216ac_x, '216')])#add jolt to FV
F216=np.append(F216,[2])
F216=np.append(F216,[jolt(dy216ac_y, '216')])#add jolt to FV
F216=np.append(F216,[2])
F216=np.append(F216,[x216time[len(x216time)-1]])
y216encoder=np.array(y216encoder)
newen=np.diff(y216encoder)
newen=np.where(newen>10, 0, newen)
F216=np.append(F216,[np.sum(newen)]) #add total number of turns
F216=np.append(F216,[len(newen[newen > 0])]) #add number of chnages in encoder
F216=np.append(F216,[len(newen[newen > 0])/x216time[len(x216time)-1]])
F216=np.append(F216,[1,216]) #label autistic
F216=np.append(F216,[jolt(dy216ac_z, '216')])#add jolt to FV
F216=np.append(F216,[2])


#strange data
print('label 217')
A217 = pd.read_csv(r'C:\PythonCode\A_parsa shamsaee.csv')   
A217=np.array(A217)
A217=np.delete(A217, [2580], 0)
A217 = np.vstack(A217[:, :]).astype(np.float)
x217time= A217[:,0]
y217ac=[A217[:,1], A217[:,2], A217[:,3]] #ac_x, ac_y, ac_z
y217encoder= [A217[:,4],A217[:,5]]  #encoder1, encoder2
plot1ac(x217time, y217ac[0],89 ,0) #see the signal before denoising
dy217ac_x=wavelet_denoising(y217ac[0])
dy217ac_y=wavelet_denoising(y217ac[1])
dy217ac_z=wavelet_denoising(y217ac[2])
#plot1ac(x217time, y217ac[0], 217, 0)
#plot1ac(x217time, dy217ac_x, 217, 1)
F217=alltop5points(dy217ac_x, dy217ac_y, dy217ac_z, x217time) #feature vector=FV
F217=np.concatenate((F217,allenergy(dy217ac_x, dy217ac_y, dy217ac_z))) #add energy to FV
F217=np.concatenate((F217,correlation(dy217ac_x, dy217ac_y, dy217ac_z))) #add correlation coefficients to FV
F217=np.append(F217,[jolt(dy217ac_x, '217')])#add jolt to FV
F217=np.append(F217,[4]) #6
F217=np.append(F217,[jolt(dy217ac_y, '217')])#add jolt to FV
F217=np.append(F217,[7])
F217=np.append(F217,[x217time[len(x217time)-1]])
y217encoder=np.array(y217encoder)
newen=np.diff(y217encoder)
newen=np.where(newen>10, 0, newen)
F217=np.append(F217,[np.sum(newen)]) #add total number of turns
F217=np.append(F217,[len(newen[newen > 0])]) #add number of chnages in encoder
F217=np.append(F217,[len(newen[newen > 0])/x217time[len(x217time)-1]])
F217=np.append(F217,[1,217]) #label autistic
F217=np.append(F217,[jolt(dy217ac_z, '217')])#add jolt to FV
F217=np.append(F217,[7])


print('label 218')
A218 = pd.read_csv(r'C:\PythonCode\A_samyar mirbeygy.csv')   
#print (A218)
A218=np.array(A218)
A218 = np.vstack(A218[:, :]).astype(np.float)
#print (A218)
x218time= A218[:,0]
y218ac=[A218[:,1], A218[:,2], A218[:,3]] #ac_x, ac_y, ac_z
y218encoder= [A218[:,4],A218[:,5]]  #encoder1, encoder2
plot1ac(x218time, y218ac[0],89 ,0) #see the signal before denoising
dy218ac_x=wavelet_denoising(y218ac[0])
dy218ac_y=wavelet_denoising(y218ac[1])
dy218ac_z=wavelet_denoising(y218ac[2])
#plot1ac(x218time, y218ac[0], 218, 0)
#plot1ac(x218time, dy218ac_x, 218, 1)
F218=alltop5points(dy218ac_x, dy218ac_y, dy218ac_z, x218time) #feature vector=FV
F218=np.concatenate((F218,allenergy(dy218ac_x, dy218ac_y, dy218ac_z))) #add energy to FV
F218=np.concatenate((F218,correlation(dy218ac_x, dy218ac_y, dy218ac_z))) #add correlation coefficients to FV
F218=np.append(F218,[jolt(dy218ac_x, '218')])#add jolt to FV
F218=np.append(F218,[3]) #4
F218=np.append(F218,[jolt(dy218ac_y, '218')])#add jolt to FV
F218=np.append(F218,[7])
F218=np.append(F218,[x218time[len(x218time)-1]])
y218encoder=np.array(y218encoder)
newen=np.diff(y218encoder)
newen=np.where(newen>10, 0, newen)
F218=np.append(F218,[np.sum(newen)]) #add total number of turns
F218=np.append(F218,[len(newen[newen > 0])]) #add number of chnages in encoder
F218=np.append(F218,[len(newen[newen > 0])/x218time[len(x218time)-1]])
F218=np.append(F218,[1,218]) #label autistic
F218=np.append(F218,[jolt(dy218ac_z, '218')])#add jolt to FV
F218=np.append(F218,[7])


print('label 301')
A301 = pd.read_csv(r'C:\PythonCode\N_Ala movahednia.csv')   
#print (A301)
A301=np.array(A301)
A301 = np.vstack(A301[:, :]).astype(np.float)
#print (A301)
x301time= A301[:,0]
y301ac=[A301[:,1], A301[:,2], A301[:,3]] #ac_x, ac_y, ac_z
y301encoder= [A301[:,4],A301[:,5]]  #encoder1, encoder2
plot1ac(x301time, y301ac[0],89 ,0) #see the signal before denoising
dy301ac_x=wavelet_denoising(y301ac[0])
dy301ac_y=wavelet_denoising(y301ac[1])
dy301ac_z=wavelet_denoising(y301ac[2])
#plot1ac(x301time, y301ac[0], 301, 0)
#plot1ac(x301time, dy301ac_x, 301, 1)
F301=alltop5points(dy301ac_x, dy301ac_y, dy301ac_z, x301time) #feature vector=FV
F301=np.concatenate((F301,allenergy(dy301ac_x, dy301ac_y, dy301ac_z))) #add energy to FV
F301=np.concatenate((F301,correlation(dy301ac_x, dy301ac_y, dy301ac_z))) #add correlation coefficients to FV
F301=np.append(F301,[jolt(dy301ac_x, '301')])#add jolt to FV
F301=np.append(F301,[5])
F301=np.append(F301,[jolt(dy301ac_y, '301')])#add jolt to FV
F301=np.append(F301,[12])
F301=np.append(F301,[x301time[len(x301time)-1]])
y301encoder=np.array(y301encoder)
newen=np.diff(y301encoder)
newen=np.where(newen>10, 0, newen)
F301=np.append(F301,[np.sum(newen)]) #add total number of turns
F301=np.append(F301,[len(newen[newen > 0])]) #add number of chnages in encoder
F301=np.append(F301,[len(newen[newen > 0])/x301time[len(x301time)-1]])
F301=np.append(F301,[0,301]) #label normal
F301=np.append(F301,[jolt(dy301ac_z, '301')])#add jolt to FV
F301=np.append(F301,[12])


print('label 302')
A302 = pd.read_csv(r'C:\PythonCode\N_Ali adibzadeh.csv')   
#print (A302)
A302=np.array(A302)
A302 = np.vstack(A302[:, :]).astype(np.float)
#print (A302)
x302time= A302[:,0]
y302ac=[A302[:,1], A302[:,2], A302[:,3]] #ac_x, ac_y, ac_z
y302encoder= [A302[:,4],A302[:,5]]  #encoder1, encoder2
plot1ac(x302time, y302ac[0],89 ,0) #see the signal before denoising
dy302ac_x=wavelet_denoising(y302ac[0])
dy302ac_y=wavelet_denoising(y302ac[1])
dy302ac_z=wavelet_denoising(y302ac[2])
#plot1ac(x302time, y302ac[0], 302, 0)
#plot1ac(x302time, dy302ac_x, 302, 1)
F302=alltop5points(dy302ac_x, dy302ac_y, dy302ac_z, x302time) #feature vector=FV
F302=np.concatenate((F302,allenergy(dy302ac_x, dy302ac_y, dy302ac_z))) #add energy to FV
F302=np.concatenate((F302,correlation(dy302ac_x, dy302ac_y, dy302ac_z))) #add correlation coefficients to FV
F302=np.append(F302,[jolt(dy302ac_x, '302')])#add jolt to FV
F302=np.append(F302,[7])
F302=np.append(F302,[jolt(dy302ac_y, '302')])#add jolt to FV
F302=np.append(F302,[14])
F302=np.append(F302,[x302time[len(x302time)-1]])
y302encoder=np.array(y302encoder)
newen=np.diff(y302encoder)
newen=np.where(newen>10, 0, newen)
F302=np.append(F302,[np.sum(newen)]) #add total number of turns
F302=np.append(F302,[len(newen[newen > 0])]) #add number of chnages in encoder
F302=np.append(F302,[len(newen[newen > 0])/x302time[len(x302time)-1]])
F302=np.append(F302,[0,302]) #label normal
F302=np.append(F302,[jolt(dy302ac_z, '302')])#add jolt to FV
F302=np.append(F302,[14])


print('label 303')
A303 = pd.read_csv(r'C:\PythonCode\N_alireza tanha.csv')   
#print (A303)
A303=np.array(A303)
A303 = np.vstack(A303[:, :]).astype(np.float)
#print (A303)
x303time= A303[:,0]
y303ac=[A303[:,1], A303[:,2], A303[:,3]] #ac_x, ac_y, ac_z
y303encoder= [A303[:,4],A303[:,5]]  #encoder1, encoder2
plot1ac(x303time, y303ac[0],89 ,0) #see the signal before denoising
dy303ac_x=wavelet_denoising(y303ac[0])
dy303ac_y=wavelet_denoising(y303ac[1])
dy303ac_z=wavelet_denoising(y303ac[2])
#plot1ac(x303time, y303ac[0], 303, 0)
#plot1ac(x303time, dy303ac_x, 303, 1)
F303=alltop5points(dy303ac_x, dy303ac_y, dy303ac_z, x303time) #feature vector=FV
F303=np.concatenate((F303,allenergy(dy303ac_x, dy303ac_y, dy303ac_z))) #add energy to FV
F303=np.concatenate((F303,correlation(dy303ac_x, dy303ac_y, dy303ac_z))) #add correlation coefficients to FV
F303=np.append(F303,[jolt(dy303ac_x, '303')])#add jolt to FV
F303=np.append(F303,[4]) #3
F303=np.append(F303,[jolt(dy303ac_y, '303')])#add jolt to FV
F303=np.append(F303,[2])
F303=np.append(F303,[x303time[len(x303time)-1]])
y303encoder=np.array(y303encoder)
newen=np.diff(y303encoder)
newen=np.where(newen>10, 0, newen)
F303=np.append(F303,[np.sum(newen)]) #add total number of turns
F303=np.append(F303,[len(newen[newen > 0])]) #add number of chnages in encoder
F303=np.append(F303,[len(newen[newen > 0])/x303time[len(x303time)-1]])
F303=np.append(F303,[0,303]) #label normal
F303=np.append(F303,[jolt(dy303ac_z, '303')])#add jolt to FV
F303=np.append(F303,[2])


print('label 304')
A304 = pd.read_csv(r'C:\PythonCode\N_amirhossein ghafarpour.csv')   
#print (A304)
A304=np.array(A304)
A304 = np.vstack(A304[:, :]).astype(np.float)
#print (A304)
x304time= A304[:,0]
y304ac=[A304[:,1], A304[:,2], A304[:,3]] #ac_x, ac_y, ac_z
y304encoder= [A304[:,4],A304[:,5]]  #encoder1, encoder2
plot1ac(x304time, y304ac[0],89 ,0) #see the signal before denoising
dy304ac_x=wavelet_denoising(y304ac[0])
dy304ac_y=wavelet_denoising(y304ac[1])
dy304ac_z=wavelet_denoising(y304ac[2])
#plot1ac(x304time, y304ac[0], 304, 0)
#plot1ac(x304time, dy304ac_x, 304, 1)
F304=alltop5points(dy304ac_x, dy304ac_y, dy304ac_z, x304time) #feature vector=FV
F304=np.concatenate((F304,allenergy(dy304ac_x, dy304ac_y, dy304ac_z))) #add energy to FV
F304=np.concatenate((F304,correlation(dy304ac_x, dy304ac_y, dy304ac_z))) #add correlation coefficients to FV
F304=np.append(F304,[jolt(dy304ac_x, '304')])#add jolt to FV
F304=np.append(F304,[3])
F304=np.append(F304,[jolt(dy304ac_y, '304')])#add jolt to FV
F304=np.append(F304,[3]) #3
F304=np.append(F304,[x304time[len(x304time)-1]])
y304encoder=np.array(y304encoder)
newen=np.diff(y304encoder)
newen=np.where(newen>10, 0, newen)
F304=np.append(F304,[np.sum(newen)]) #add total number of turns
F304=np.append(F304,[len(newen[newen > 0])]) #add number of chnages in encoder
F304=np.append(F304,[len(newen[newen > 0])/x304time[len(x304time)-1]])
F304=np.append(F304,[0,304]) #label normal
F304=np.append(F304,[jolt(dy304ac_z, '304')])#add jolt to FV
F304=np.append(F304,[3]) #3


print('label 305')
A305 = pd.read_csv(r'C:\PythonCode\N_bahar erfanian.csv')   
#print (A305)
A305=np.array(A305)
A305 = np.vstack(A305[:, :]).astype(np.float)
#print (A305)
x305time= A305[:,0]
y305ac=[A305[:,1], A305[:,2], A305[:,3]] #ac_x, ac_y, ac_z
y305encoder= [A305[:,4],A305[:,5]]  #encoder1, encoder2
plot1ac(x305time, y305ac[0],89 ,0) #see the signal before denoising
dy305ac_x=wavelet_denoising(y305ac[0])
dy305ac_y=wavelet_denoising(y305ac[1])
dy305ac_z=wavelet_denoising(y305ac[2])
#plot1ac(x305time, y305ac[0], 305, 0)
#plot1ac(x305time, dy305ac_x, 305, 1)
F305=alltop5points(dy305ac_x, dy305ac_y, dy305ac_z, x305time) #feature vector=FV
F305=np.concatenate((F305,allenergy(dy305ac_x, dy305ac_y, dy305ac_z))) #add energy to FV
F305=np.concatenate((F305,correlation(dy305ac_x, dy305ac_y, dy305ac_z))) #add correlation coefficients to FV
F305=np.append(F305,[jolt(dy305ac_x, '305')])#add jolt to FV
F305=np.append(F305,[7]) #8
F305=np.append(F305,[jolt(dy305ac_y, '305')])#add jolt to FV
F305=np.append(F305,[14])
F305=np.append(F305,[x305time[len(x305time)-1]])
y305encoder=np.array(y305encoder)
newen=np.diff(y305encoder)
newen=np.where(newen>10, 0, newen)
F305=np.append(F305,[np.sum(newen)]) #add total number of turns
F305=np.append(F305,[len(newen[newen > 0])]) #add number of chnages in encoder
F305=np.append(F305,[len(newen[newen > 0])/x305time[len(x305time)-1]])
F305=np.append(F305,[0,305]) #label normal
F305=np.append(F305,[jolt(dy305ac_z, '305')])#add jolt to FV
F305=np.append(F305,[14])


print('label 306')
A306 = pd.read_csv(r'C:\PythonCode\N_dorsa ebrahimi.csv')   
#print (A306)
A306=np.array(A306)
A306 = np.vstack(A306[:, :]).astype(np.float)
#print (A306)
x306time= A306[:,0]
y306ac=[A306[:,1], A306[:,2], A306[:,3]] #ac_x, ac_y, ac_z
y306encoder= [A306[:,4],A306[:,5]]  #encoder1, encoder2
plot1ac(x306time, y306ac[0],89 ,0) #see the signal before denoising
dy306ac_x=wavelet_denoising(y306ac[0])
dy306ac_y=wavelet_denoising(y306ac[1])
dy306ac_z=wavelet_denoising(y306ac[2])
#plot1ac(x306time, y306ac[0], 306, 0)
#plot1ac(x306time, dy306ac_x, 306, 1)
F306=alltop5points(dy306ac_x, dy306ac_y, dy306ac_z, x306time) #feature vector=FV
F306=np.concatenate((F306,allenergy(dy306ac_x, dy306ac_y, dy306ac_z))) #add energy to FV
F306=np.concatenate((F306,correlation(dy306ac_x, dy306ac_y, dy306ac_z))) #add correlation coefficients to FV
F306=np.append(F306,[jolt(dy306ac_x, '306')])#add jolt to FV
F306=np.append(F306,[5])
F306=np.append(F306,[jolt(dy306ac_y, '306')])#add jolt to FV
F306=np.append(F306,[7])
F306=np.append(F306,[x306time[len(x306time)-1]])
y306encoder=np.array(y306encoder)
newen=np.diff(y306encoder)
newen=np.where(newen>10, 0, newen)
F306=np.append(F306,[np.sum(newen)]) #add total number of turns
F306=np.append(F306,[len(newen[newen > 0])]) #add number of chnages in encoder
F306=np.append(F306,[len(newen[newen > 0])/x306time[len(x306time)-1]])
F306=np.append(F306,[0,306]) #label normal
F306=np.append(F306,[jolt(dy306ac_z, '306')])#add jolt to FV
F306=np.append(F306,[7])


print('label 307')
A307 = pd.read_csv(r'C:\PythonCode\N_fatemeh sadat falah.csv')   
#print (A307)
A307=np.array(A307)
A307 = np.vstack(A307[:, :]).astype(np.float)
#print (A307)
x307time= A307[:,0]
y307ac=[A307[:,1], A307[:,2], A307[:,3]] #ac_x, ac_y, ac_z
y307encoder= [A307[:,4],A307[:,5]]  #encoder1, encoder2
plot1ac(x307time, y307ac[0],89 ,0) #see the signal before denoising
dy307ac_x=wavelet_denoising(y307ac[0])
dy307ac_y=wavelet_denoising(y307ac[1])
dy307ac_z=wavelet_denoising(y307ac[2])
#plot1ac(x307time, y307ac[0], 307, 0)
#plot1ac(x307time, dy307ac_x, 307, 1)
F307=alltop5points(dy307ac_x, dy307ac_y, dy307ac_z, x307time) #feature vector=FV
F307=np.concatenate((F307,allenergy(dy307ac_x, dy307ac_y, dy307ac_z))) #add energy to FV
F307=np.concatenate((F307,correlation(dy307ac_x, dy307ac_y, dy307ac_z))) #add correlation coefficients to FV
F307=np.append(F307,[jolt(dy307ac_x, '307')])#add jolt to FV
F307=np.append(F307,[7])
F307=np.append(F307,[jolt(dy307ac_y, '307')])#add jolt to FV
F307=np.append(F307,[8])
F307=np.append(F307,[x307time[len(x307time)-1]])
y307encoder=np.array(y307encoder)
newen=np.diff(y307encoder)
newen=np.where(newen>10, 0, newen)
F307=np.append(F307,[np.sum(newen)]) #add total number of turns
F307=np.append(F307,[len(newen[newen > 0])]) #add number of chnages in encoder
F307=np.append(F307,[len(newen[newen > 0])/x307time[len(x307time)-1]])
F307=np.append(F307,[0,307]) #label normal
F307=np.append(F307,[jolt(dy307ac_z, '307')])#add jolt to FV
F307=np.append(F307,[8])


print('label 308')
A308 = pd.read_csv(r'C:\PythonCode\N_hosein aminian.csv')   
#print (A308)
A308=np.array(A308)
A308 = np.vstack(A308[:, :]).astype(np.float)
#print (A308)
x308time= A308[:,0]
y308ac=[A308[:,1], A308[:,2], A308[:,3]] #ac_x, ac_y, ac_z
y308encoder= [A308[:,4],A308[:,5]]  #encoder1, encoder2
plot1ac(x308time, y308ac[0],89 ,0) #see the signal before denoising
dy308ac_x=wavelet_denoising(y308ac[0])
dy308ac_y=wavelet_denoising(y308ac[1])
dy308ac_z=wavelet_denoising(y308ac[2])
#plot1ac(x308time, y308ac[0], 308, 0)
#plot1ac(x308time, dy308ac_x, 308, 1)
F308=alltop5points(dy308ac_x, dy308ac_y, dy308ac_z, x308time) #feature vector=FV
F308=np.concatenate((F308,allenergy(dy308ac_x, dy308ac_y, dy308ac_z))) #add energy to FV
F308=np.concatenate((F308,correlation(dy308ac_x, dy308ac_y, dy308ac_z))) #add correlation coefficients to FV
F308=np.append(F308,[jolt(dy308ac_x, '308')])#add jolt to FV
F308=np.append(F308,[4]) #7
F308=np.append(F308,[jolt(dy308ac_y, '308')])#add jolt to FV
F308=np.append(F308,[5])
F308=np.append(F308,[x308time[len(x308time)-1]])
y308encoder=np.array(y308encoder)
newen=np.diff(y308encoder)
newen=np.where(newen>10, 0, newen)
F308=np.append(F308,[np.sum(newen)]) #add total number of turns
F308=np.append(F308,[len(newen[newen > 0])]) #add number of chnages in encoder
F308=np.append(F308,[len(newen[newen > 0])/x308time[len(x308time)-1]])
F308=np.append(F308,[0,308]) #label normal
F308=np.append(F308,[jolt(dy308ac_z, '308')])#add jolt to FV
F308=np.append(F308,[5])


print('label 309')
A309 = pd.read_csv(r'C:\PythonCode\N_hosna golshenas.csv')   
#print (A309)
A309=np.array(A309)
A309 = np.vstack(A309[:, :]).astype(np.float)
#print (A309)
x309time= A309[:,0]
y309ac=[A309[:,1], A309[:,2], A309[:,3]] #ac_x, ac_y, ac_z
y309encoder= [A309[:,4],A309[:,5]]  #encoder1, encoder2
plot1ac(x309time, y309ac[0],89 ,0) #see the signal before denoising
dy309ac_x=wavelet_denoising(y309ac[0])
dy309ac_y=wavelet_denoising(y309ac[1])
dy309ac_z=wavelet_denoising(y309ac[2])
#plot1ac(x309time, y309ac[0], 309, 0)
#plot1ac(x309time, dy309ac_x, 309, 1)
F309=alltop5points(dy309ac_x, dy309ac_y, dy309ac_z, x309time) #feature vector=FV
F309=np.concatenate((F309,allenergy(dy309ac_x, dy309ac_y, dy309ac_z))) #add energy to FV
F309=np.concatenate((F309,correlation(dy309ac_x, dy309ac_y, dy309ac_z))) #add correlation coefficients to FV
F309=np.append(F309,[jolt(dy309ac_x, '309')])#add jolt to FV
F309=np.append(F309,[4]) #5
F309=np.append(F309,[jolt(dy309ac_y, '309')])#add jolt to FV
F309=np.append(F309,[13])
F309=np.append(F309,[x309time[len(x309time)-1]])
y309encoder=np.array(y309encoder)
newen=np.diff(y309encoder)
newen=np.where(newen>10, 0, newen)
F309=np.append(F309,[np.sum(newen)]) #add total number of turns
F309=np.append(F309,[len(newen[newen > 0])]) #add number of chnages in encoder
F309=np.append(F309,[len(newen[newen > 0])/x309time[len(x309time)-1]])
F309=np.append(F309,[0,309]) #label normal
F309=np.append(F309,[jolt(dy309ac_z, '309')])#add jolt to FV
F309=np.append(F309,[13])


print('label 310')
A310 = pd.read_csv(r'C:\PythonCode\N_hosna karimi.csv')   
#print (A310)
A310=np.array(A310)
A310 = np.vstack(A310[:, :]).astype(np.float)
#print (A310)
x310time= A310[:,0]
y310ac=[A310[:,1], A310[:,2], A310[:,3]] #ac_x, ac_y, ac_z
y310encoder= [A310[:,4],A310[:,5]]  #encoder1, encoder2
plot1ac(x310time, y310ac[0],89 ,0) #see the signal before denoising
dy310ac_x=wavelet_denoising(y310ac[0])
dy310ac_y=wavelet_denoising(y310ac[1])
dy310ac_z=wavelet_denoising(y310ac[2])
#plot1ac(x310time, y310ac[0], 310, 0)
#plot1ac(x310time, dy310ac_x, 310, 1)
F310=alltop5points(dy310ac_x, dy310ac_y, dy310ac_z, x310time) #feature vector=FV
F310=np.concatenate((F310,allenergy(dy310ac_x, dy310ac_y, dy310ac_z))) #add energy to FV
F310=np.concatenate((F310,correlation(dy310ac_x, dy310ac_y, dy310ac_z))) #add correlation coefficients to FV
F310=np.append(F310,[jolt(dy310ac_x, '310')])#add jolt to FV
F310=np.append(F310,[3])
F310=np.append(F310,[jolt(dy310ac_y, '310')])#add jolt to FV
F310=np.append(F310,[3])
F310=np.append(F310,[x310time[len(x310time)-1]])
y310encoder=np.array(y310encoder)
newen=np.diff(y310encoder)
newen=np.where(newen>10, 0, newen)
F310=np.append(F310,[np.sum(newen)]) #add total number of turns
F310=np.append(F310,[len(newen[newen > 0])]) #add number of chnages in encoder
F310=np.append(F310,[len(newen[newen > 0])/x310time[len(x310time)-1]])
F310=np.append(F310,[0,310]) #label normal
F310=np.append(F310,[jolt(dy310ac_z, '310')])#add jolt to FV
F310=np.append(F310,[3])


print('label 311')
A311 = pd.read_csv(r'C:\PythonCode\N_mohammad amin salem.csv')   
#print (A311)
A311=np.array(A311)
A311 = np.vstack(A311[:, :]).astype(np.float)
#print (A311)
x311time= A311[:,0]
y311ac=[A311[:,1], A311[:,2], A311[:,3]] #ac_x, ac_y, ac_z
y311encoder= [A311[:,4],A311[:,5]]  #encoder1, encoder2
plot1ac(x311time, y311ac[0],89 ,0) #see the signal before denoising
dy311ac_x=wavelet_denoising(y311ac[0])
dy311ac_y=wavelet_denoising(y311ac[1])
dy311ac_z=wavelet_denoising(y311ac[2])
#plot1ac(x311time, y311ac[0], 311, 0)
#plot1ac(x311time, dy311ac_x, 311, 1)
F311=alltop5points(dy311ac_x, dy311ac_y, dy311ac_z, x311time) #feature vector=FV
F311=np.concatenate((F311,allenergy(dy311ac_x, dy311ac_y, dy311ac_z))) #add energy to FV
F311=np.concatenate((F311,correlation(dy311ac_x, dy311ac_y, dy311ac_z))) #add correlation coefficients to FV
F311=np.append(F311,[jolt(dy311ac_x, '311')])#add jolt to FV
F311=np.append(F311,[2])
F311=np.append(F311,[jolt(dy311ac_y, '311')])#add jolt to FV
F311=np.append(F311,[2])
F311=np.append(F311,[x311time[len(x311time)-1]])
y311encoder=np.array(y311encoder)
newen=np.diff(y311encoder)
newen=np.where(newen>10, 0, newen)
F311=np.append(F311,[np.sum(newen)]) #add total number of turns
F311=np.append(F311,[len(newen[newen > 0])]) #add number of chnages in encoder
F311=np.append(F311,[len(newen[newen > 0])/x311time[len(x311time)-1]])
F311=np.append(F311,[0,311]) #label normal
F311=np.append(F311,[jolt(dy311ac_z, '311')])#add jolt to FV
F311=np.append(F311,[2])


print('label 312')
A312 = pd.read_csv(r'C:\PythonCode\N_mohammad amir ghobadi.csv')   
#print (A312)
A312=np.array(A312)
A312 = np.vstack(A312[:, :]).astype(np.float)
#print (A312)
x312time= A312[:,0]
y312ac=[A312[:,1], A312[:,2], A312[:,3]] #ac_x, ac_y, ac_z
y312encoder= [A312[:,4],A312[:,5]]  #encoder1, encoder2
plot1ac(x312time, y312ac[0],89 ,0) #see the signal before denoising
dy312ac_x=wavelet_denoising(y312ac[0])
dy312ac_y=wavelet_denoising(y312ac[1])
dy312ac_z=wavelet_denoising(y312ac[2])
#plot1ac(x312time, y312ac[0], 312, 0)
#plot1ac(x312time, dy312ac_x, 312, 1)
F312=alltop5points(dy312ac_x, dy312ac_y, dy312ac_z, x312time) #feature vector=FV
F312=np.concatenate((F312,allenergy(dy312ac_x, dy312ac_y, dy312ac_z))) #add energy to FV
F312=np.concatenate((F312,correlation(dy312ac_x, dy312ac_y, dy312ac_z))) #add correlation coefficients to FV
F312=np.append(F312,[jolt(dy312ac_x, '312')])#add jolt to FV
F312=np.append(F312,[7]) #6
F312=np.append(F312,[jolt(dy312ac_y, '312')])#add jolt to FV
F312=np.append(F312,[3]) 
F312=np.append(F312,[x312time[len(x312time)-1]])
y312encoder=np.array(y312encoder)
newen=np.diff(y312encoder)
newen=np.where(newen>10, 0, newen)
F312=np.append(F312,[np.sum(newen)]) #add total number of turns
F312=np.append(F312,[len(newen[newen > 0])]) #add number of chnages in encoder
F312=np.append(F312,[len(newen[newen > 0])/x312time[len(x312time)-1]])
F312=np.append(F312,[0,312]) #label normal
F312=np.append(F312,[jolt(dy312ac_z, '312')])#add jolt to FV
F312=np.append(F312,[3]) 


print('label 313')
A313 = pd.read_csv(r'C:\PythonCode\N_mohammad sadra khayami.csv')   
#print (A313)
A313=np.array(A313)
A313 = np.vstack(A313[:, :]).astype(np.float)
#print (A313)
x313time= A313[:,0]
y313ac=[A313[:,1], A313[:,2], A313[:,3]] #ac_x, ac_y, ac_z
y313encoder= [A313[:,4],A313[:,5]]  #encoder1, encoder2
plot1ac(x313time, y313ac[0],89 ,0) #see the signal before denoising
dy313ac_x=wavelet_denoising(y313ac[0])
dy313ac_y=wavelet_denoising(y313ac[1])
dy313ac_z=wavelet_denoising(y313ac[2])
#plot1ac(x313time, y313ac[0], 313, 0)
#plot1ac(x313time, dy313ac_x, 313, 1)
F313=alltop5points(dy313ac_x, dy313ac_y, dy313ac_z, x313time) #feature vector=FV
F313=np.concatenate((F313,allenergy(dy313ac_x, dy313ac_y, dy313ac_z))) #add energy to FV
F313=np.concatenate((F313,correlation(dy313ac_x, dy313ac_y, dy313ac_z))) #add correlation coefficients to FV
F313=np.append(F313,[jolt(dy313ac_x, '313')])#add x jolt to FV
F313=np.append(F313,[8]) #9
F313=np.append(F313,[jolt(dy313ac_y, '313')])#add y jolt to FV
F313=np.append(F313,[7])
F313=np.append(F313,[x313time[len(x313time)-1]])
y313encoder=np.array(y313encoder)
newen=np.diff(y313encoder)
newen=np.where(newen>10, 0, newen)
F313=np.append(F313,[np.sum(newen)]) #add total number of turns
F313=np.append(F313,[len(newen[newen > 0])]) #add number of chnages in encoder
F313=np.append(F313,[len(newen[newen > 0])/x313time[len(x313time)-1]])
F313=np.append(F313,[0,313]) #label normal
F313=np.append(F313,[jolt(dy313ac_z, '313')])#add y jolt to FV
F313=np.append(F313,[7])


print('label 314')
A314 = pd.read_csv(r'C:\PythonCode\N_sobhan aghamohammad.csv')   
#print (A314)
A314=np.array(A314)
A314 = np.vstack(A314[:, :]).astype(np.float)
#print (A314)
x314time= A314[:,0]
y314ac=[A314[:,1], A314[:,2], A314[:,3]] #ac_x, ac_y, ac_z
y314encoder= [A314[:,4],A314[:,5]]  #encoder1, encoder2
plot1ac(x314time, y314ac[0],89 ,0) #see the signal before denoising
dy314ac_x=wavelet_denoising(y314ac[0])
dy314ac_y=wavelet_denoising(y314ac[1])
dy314ac_z=wavelet_denoising(y314ac[2])
#plot1ac(x314time, y314ac[0], 314, 0)
#plot1ac(x314time, dy314ac_x, 314, 1)
F314=alltop5points(dy314ac_x, dy314ac_y, dy314ac_z, x314time) #feature vector=FV
F314=np.concatenate((F314,allenergy(dy314ac_x, dy314ac_y, dy314ac_z))) #add energy to FV
F314=np.concatenate((F314,correlation(dy314ac_x, dy314ac_y, dy314ac_z))) #add correlation coefficients to FV
F314=np.append(F314,[jolt(dy314ac_x, '314')])#add jolt to FV
F314=np.append(F314,[1])
F314=np.append(F314,[jolt(dy314ac_y, '314')])#add jolt to FV
F314=np.append(F314,[3])
F314=np.append(F314,[x314time[len(x314time)-1]])
y314encoder=np.array(y314encoder)
newen=np.diff(y314encoder)
newen=np.where(newen>10, 0, newen)
F314=np.append(F314,[np.sum(newen)]) #add total number of turns
F314=np.append(F314,[len(newen[newen > 0])]) #add number of chnages in encoder
F314=np.append(F314,[len(newen[newen > 0])/x314time[len(x314time)-1]])
F314=np.append(F314,[0,314]) #label normal
F314=np.append(F314,[jolt(dy314ac_z, '314')])#add jolt to FV
F314=np.append(F314,[3])


print('label 315')
A315 = pd.read_csv(r'C:\PythonCode\N_zahra tavakoli.csv')   
#print (A315)
A315=np.array(A315)
A315 = np.vstack(A315[:, :]).astype(np.float)
#print (A315)
x315time= A315[:,0]
y315ac=[A315[:,1], A315[:,2], A315[:,3]] #ac_x, ac_y, ac_z
y315encoder= [A315[:,4],A315[:,5]]  #encoder1, encoder2
plot1ac(x315time, y315ac[0],89 ,0) #see the signal before denoising
dy315ac_x=wavelet_denoising(y315ac[0])
dy315ac_y=wavelet_denoising(y315ac[1])
dy315ac_z=wavelet_denoising(y315ac[2])
#plot1ac(x315time, y315ac[0], 315, 0)
#plot1ac(x315time, dy315ac_x, 315, 1)
F315=alltop5points(dy315ac_x, dy315ac_y, dy315ac_z, x315time) #feature vector=FV
F315=np.concatenate((F315,allenergy(dy315ac_x, dy315ac_y, dy315ac_z))) #add energy to FV
F315=np.concatenate((F315,correlation(dy315ac_x, dy315ac_y, dy315ac_z))) #add correlation coefficients to FV
F315=np.append(F315,[jolt(dy315ac_x, '315')])#add jolt to FV
F315=np.append(F315,[4]) #5
F315=np.append(F315,[jolt(dy315ac_y, '315')])#add jolt to FV
F315=np.append(F315,[9]) #5
F315=np.append(F315,[x315time[len(x315time)-1]])
y315encoder=np.array(y315encoder)
newen=np.diff(y315encoder)
newen=np.where(newen>10, 0, newen)
F315=np.append(F315,[np.sum(newen)]) #add total number of turns
F315=np.append(F315,[len(newen[newen > 0])]) #add number of chnages in encoder
F315=np.append(F315,[len(newen[newen > 0])/x315time[len(x315time)-1]])
F315=np.append(F315,[0,315]) #label normal
F315=np.append(F315,[jolt(dy315ac_z, '315')])#add jolt to FV
F315=np.append(F315,[9]) #5


# info=np.array([1 ,1 ,1 ,1 ,1 ,0,0,0,'NA108','NA120','NA121','NA124'])
#info1=np.array(5*['fx']+5*['fy']+5*['fz']+5*['Ax']+5*['Ay']+5*['Az']+['Ex']+['Ey']+['Ez']+['Cxy']+['Cxz']+['Cyz']+['J'])
#info2=np.array(['0','au95','au100','au106','au112','au118','no89','no94','no102','NA108','NA120','NA121','NA124'])
Fautistic=np.concatenate((F95.reshape(1,46),F100.reshape(1,46),F106.reshape(1,46),F112.reshape(1,46),F118.reshape(1,46),F201.reshape(1,46),F202.reshape(1,46),F203.reshape(1,46),F204.reshape(1,46),F205.reshape(1,46),F206.reshape(1,46),F207.reshape(1,46),F208.reshape(1,46),F209.reshape(1,46),F210.reshape(1,46),F211.reshape(1,46),F212.reshape(1,46),F213.reshape(1,46),F214.reshape(1,46),F215.reshape(1,46),F216.reshape(1,46),F217.reshape(1,46),F218.reshape(1,46)))
Fnormal=np.concatenate((F89.reshape(1,46),F94.reshape(1,46),F102.reshape(1,46),F301.reshape(1,46),F302.reshape(1,46),F303.reshape(1,46),F304.reshape(1,46),F305.reshape(1,46),F306.reshape(1,46),F307.reshape(1,46),F308.reshape(1,46),F309.reshape(1,46),F310.reshape(1,46),F311.reshape(1,46),F312.reshape(1,46),F313.reshape(1,46),F314.reshape(1,46),F315.reshape(1,46)))
Fothers=np.concatenate((F108.reshape(1,46),F120.reshape(1,46),F121.reshape(1,46),F124.reshape(1,46)))
#Ftotal=np.concatenate((info1.reshape(1,37),Fautistic,Fnormal,Fothers))
Ftotal=np.concatenate((Fautistic,Fnormal,Fothers))
Ftotal=np.transpose(Ftotal)
#Ftotal=np.concatenate((info2.reshape(1,13),Ftotal))
print('Ftotal')
print(Ftotal)

# np.savetxt('features.csv', Ftotal)
# np.save('features.csv', Ftotal)

# Ftotal=np.transpose(Ftotal)
# X= np.delete(Ftotal, 41, 1) #remove label from other features
# y= Ftotal[:, 41] #label of set

# #find features with high correlation to remove them
# df = pd.DataFrame(X)
# print('df.head()=') 
# print(df.head())    
# # Create correlation matrix
# corr_matrix = df.corr().abs()
# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# # Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# # Drop features 
# df.drop(to_drop, axis=1, inplace=True)
# print()
# print('to_drop=')
# print(to_drop)
# #error!
# # df1 = df.drop(df.columns[to_drop], axis=1)
# # print()
# # print('to_drop=')
# # print(df1.head())

# # #finding k best features
# # from sklearn.feature_selection import SelectKBest
# # from sklearn.feature_selection import chi2
# # #Ftotal=np.transpose(Ftotal) #transposed earlier
# # #X= np.delete(Ftotal, 38, 1) #removed label earlier
# # #y= Ftotal[:, 38] #removed label earlier
# # X_new1 = SelectKBest(chi2, k=5).fit_transform(np.absolute(X), y)
# # #X_new.shape
# # print(X_new1)

# Ftotal=np.transpose(Ftotal)

# #SVM with 5 features
# Fselect= np.array([Ftotal[4,:],Ftotal[14,:],Ftotal[30,:],Ftotal[33,:],Ftotal[37,:],Ftotal[41,:]]) #choosing features+labels as input of SVM
# Ftrain=np.delete(Fselect, [0,1,6,13,17,19,24,28,32,37,38,49,50,51,52,53,54,55,56], 1)
# Ftrain=np.transpose(Ftrain)
# Ftest=np.delete(Fselect, [2,3,4,5,7,8,9,10,11,12,14,15,16,18,20,21,22,23,25,26,27,29,30,31,33,34,35,36,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56], 1)
# Ftest=np.transpose(Ftest)

# x_train= np.delete(Ftrain, [5], 1) #remove label
# y_train= Ftrain[:,5] #select label
# x_test= np.delete(Ftest, [5], 1) #remove label
# y_test= Ftest[:,5]

# svclassifier = sk.SVC(kernel='linear') #linear, rbf, or poly
# svclassifier.fit(x_train, y_train)
# y_pred= svclassifier.predict(x_test)

# from sklearn.metrics import classification_report, confusion_matrix
# print('with 5 features')
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# #SVM with 5 features
# Fselect= np.array([Ftotal[4,:],Ftotal[38,:],Ftotal[30,:],Ftotal[33,:],Ftotal[37,:],Ftotal[41,:]]) #choosing features+labels as input of SVM
# Ftrain=np.delete(Fselect, [0,1,6,13,17,19,24,28,32,37,38,49,50,51,52,53,54,55,56], 1)
# Ftrain=np.transpose(Ftrain)
# Ftest=np.delete(Fselect, [2,3,4,5,7,8,9,10,11,12,14,15,16,18,20,21,22,23,25,26,27,29,30,31,33,34,35,36,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56], 1)
# Ftest=np.transpose(Ftest)

# x_train= np.delete(Ftrain, [5], 1) #remove label
# y_train= Ftrain[:,5] #select label
# x_test= np.delete(Ftest, [5], 1) #remove label
# y_test= Ftest[:,5]

# svclassifier = sk.SVC(kernel='linear') #linear, rbf, or poly
# svclassifier.fit(x_train, y_train)
# y_pred= svclassifier.predict(x_test)

# from sklearn.metrics import classification_report, confusion_matrix
# print('with 5 features+encoder')
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))


# #SVM with 3 features
# Fselect= np.array([Ftotal[14,:],Ftotal[33,:],Ftotal[37,:],Ftotal[41,:]])
# Ftrain=np.delete(Fselect, [0,1,6,13,17,19,24,28,32,37,38,49,50,51,52,53,54,55,56], 1)
# Ftrain=np.transpose(Ftrain)
# Ftest=np.delete(Fselect, [2,3,4,5,7,8,9,10,11,12,14,15,16,18,20,21,22,23,25,26,27,29,30,31,33,34,35,36,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56], 1)
# Ftest=np.transpose(Ftest)

# x_train= np.delete(Ftrain, [3], 1) #remove label
# y_train= Ftrain[:,3] #select label
# x_test= np.delete(Ftest, [3], 1) #remove label
# y_test= Ftest[:,3]

# svclassifier = sk.SVC(kernel='linear') #linear, rbf, or poly
# svclassifier.fit(x_train, y_train)
# y_pred= svclassifier.predict(x_test)

# from sklearn.metrics import classification_report, confusion_matrix
# print('with 3 features')
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))


# #plot 3D for 3 features
# from sklearn.svm import SVC
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets
# from mpl_toolkits.mplot3d import Axes3D

# Fselect= np.array([Ftotal[38,:],Ftotal[39,:],Ftotal[40,:],Ftotal[42,:]])
# #Ftrain=np.delete(Fselect, [0,1,6,13,17,19,24,28,32,37,38,49,50,51,52,53,54,55,56], 1)
# Ftrain=np.transpose(Fselect)
# x_train= np.delete(Ftrain, [3], 1) #remove label
# y_train= Ftrain[:,3] #select label

# #iris = datasets.load_iris()
# X = x_train  # we only take the first three features.
# Y = y_train
# #make it binary classification problem
# X = X[np.logical_or(Y==0,Y==1)]
# Y = Y[np.logical_or(Y==0,Y==1)]

# model = svm.SVC(kernel='linear')
# clf = model.fit(X, Y)
# # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# # Solve for w3 (z)
# z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

# tmp = np.linspace(-5,5,30)
# x,y = np.meshgrid(tmp,tmp)

# fig = plt.figure()
# ax  = fig.add_subplot(111, projection='3d')
# ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
# ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
# ax.plot_surface(x, y, z(x,y))
# ax.view_init(30, 60)
# plt.show()

# # Fselect= np.array([Ftotal[14,:],Ftotal[33,:],Ftotal[38,:]])
# # Ftrain=np.delete(Fselect, [3,4,7,8], 1)
# # Ftrain=np.transpose(Ftrain)
# # Ftest=np.delete(Fselect, [0,1,2,5,6,9,10,11], 1)
# # Ftest=np.transpose(Ftest)

# # x_train= np.delete(Ftrain, [2], 1) #remove label
# # y_train= Ftrain[:,2] #select label
# # x_test= np.delete(Ftest, [2], 1) #remove label
# # y_test= Ftest[:,2]

# # svclassifier = sk.SVC(kernel='rbf')
# # svclassifier.fit(x_train, y_train)

# # y_pred= svclassifier.predict(x_test)

# # from sklearn.metrics import classification_report, confusion_matrix
# # print('with 2 features-rbf')
# # print(confusion_matrix(y_test,y_pred))
# # print(classification_report(y_test,y_pred))

# # # from mlxtend.evaluate import bias_variance_decomp
# # # model=y_pred
# # # mse, bias, var = bias_variance_decomp(model, x_train, y_train, x_test, y_test)
# # # # summarize results
# # # print('MSE: %.3f' % mse)
# # # print('Bias: %.3f' % bias)
# # # print('Variance: %.3f' % var)

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn import svm, datasets
# # # import some data to play with
# # #iris = datasets.load_iris()
# # X = x_train  # we only take the first two features. We could                    
# # y = y_train

# # h = .02  # step size in the mesh

# # # we create an instance of SVM and fit out data. We do not scale our
# # # data since we want to plot the support vectors
# # C = 1.0  # SVM regularization parameter
# # svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# # rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# # poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
# # lin_svc = svm.LinearSVC(C=C).fit(X, y)

# # # create a mesh to plot in
# # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
# #                       np.arange(y_min, y_max, h))

# # # title for the plots
# # titles = ['SVC with linear kernel',
# #           'LinearSVC (linear kernel)',
# #           'SVC with RBF kernel',
# #           'SVC with polynomial (degree 3) kernel']

# # for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
# #     # Plot the decision boundary. For that, we will assign a color to each
# #     # point in the mesh [x_min, x_max]x[y_min, y_max].
# #     plt.subplot(2, 2, i + 1)
# #     plt.subplots_adjust(wspace=0.4, hspace=0.4)

# #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# #     # Put the result into a color plot
# #     Z = Z.reshape(xx.shape)
# #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# #     # Plot also the training points
# #     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
# #     plt.xlabel('Sepal length')
# #     plt.ylabel('Sepal width')
# #     plt.xlim(xx.min(), xx.max())
# #     plt.ylim(yy.min(), yy.max())
# #     plt.xticks(())
# #     plt.yticks(())
# #     plt.title(titles[i])

# # plt.show()

# # #mp.ver
# # # a=Ftotal[30,:]
# # # b=Ftotal[36,:]
# # # plt.scatter(a[0:5], b[0:5])
# # # plt.scatter(a[5:8], b[5:8])
# # # plt.scatter(a[8:12], b[8:12])
# # # plt.xlabel("Ex")
# # # plt.ylabel("J") 
# # # plt.legend() 
# # # plt.show()



