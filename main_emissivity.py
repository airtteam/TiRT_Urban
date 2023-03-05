# This is a sample Python script.
import matplotlib.pyplot as plt

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from rt.simple_urban_laa_waa import *
from utils.utils import *
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    urban = Urban()
    wl = 10.5
    sza = 20
    saa = 0
    vza_1 = np.arange(60,0,-5)
    vza_2 = np.arange(0,61,5)
    vza_ = np.hstack([vza_1,vza_2])
    # vza_ = np.hstack([vza_2])
    n_angle1 = np.size(vza_1)
    n_angle2 = np.size(vza_2)

    vaa_ = np.hstack([np.repeat(180,n_angle1),np.repeat(0,n_angle2)])
    # vaa_ = np.hstack([np.repeat(0,n_angle2)])
    Estreat = 0.97
    Eroof = 0.94
    Ewall = 0.88
    Twall_sunlit = 45 + 273.15
    Twall_shaded = 30+ 273.15
    Tstreet_sunlit = 45+ 273.15
    Tstreet_shaded = 30+ 273.15
    Troof_sunlit = 45+ 273.15
    Troof_shaded = 30+ 273.15

    Bwall_sunlit = planck(wl,Twall_sunlit)
    Bwall_shaded = planck(wl,Twall_shaded)
    Bstreet_sunlit = planck(wl,Tstreet_sunlit)
    Bstreet_shaded = planck(wl,Tstreet_shaded)
    Broof_sunlit = planck(wl,Troof_sunlit)
    Broof_shaded = planck(wl,Troof_shaded)
    B_ = np.asarray([Bwall_sunlit,Bwall_shaded,Bstreet_sunlit,
                     Bstreet_shaded,Broof_sunlit,Broof_shaded])
    shapes = np.asarray([[10, 10, 20, 0.003*0.333,0,90],
                         [10, 10, 30, 0.003*0.333,0,90],
                         [10, 10, 40, 0.003*0.333,0,90]])



    urban.set_angular_input(vza_,vaa_,sza,saa)
    urban.set_strcutural_input(shapes)
    urban.set_spectral_input(Estreat,Ewall,Eroof)
    # emissivity_1 = urban.calculate_effective_component_emissivity(1)
    #

    # #
    # plt.plot(vza_,emissivity_1,'o-')
    # # plt.plot(vza_,emissivity_2,'k')
    # plt.xlabel('VZA')
    # plt.ylabel('Effective Emissivity')
    # plt.legend(['Wall','Street','Roof'])
    # plt.show()

    # emissivity_11 = np.sum(emissivity_1,axis=1)
    # plt.figure(figsize=[4.5,4])
    # plt.plot(vza_, emissivity_11, 'o-')
    # plt.xlabel('VZA')
    # plt.ylabel('Effective Emissivity')
    # plt.show()
    # print(emissivity_1)


    emissivity_1 = urban.calculate_effective_component_emissivity(1)
    lse = np.sum(emissivity_1,axis=1)
    plt.plot(lse)
    plt.xlabel('VZA')
    plt.ylabel('Brightness Temperatures')


    height = 20 * 0.33 + 30*0.33 + 40*0.33
    shapes = np.asarray([[10,10,height,0.003,0,90]])
    urban.set_angular_input(vza_,vaa_,sza,saa)
    urban.set_strcutural_input(shapes)
    urban.set_spectral_input(Estreat,Ewall,Eroof)

    emissivity_1 = urban.calculate_effective_component_emissivity(1)
    lse = np.sum(emissivity_1,axis=1)
    plt.plot(lse)
    plt.xlabel('VZA')
    plt.ylabel('Brightness Temperatures')
    plt.legend(['het','hom'])
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
