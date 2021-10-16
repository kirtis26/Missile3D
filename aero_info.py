import numpy as np
import pandas as pd

def table_atm(h, parametr):
    """
    Cтандартная атмосфера для высот h = -2000 м ... 80000 м (ГОСТ 4401-81)
    arguments: h высота [м], parametr:
                             1 - температура [К];
                             2 - давление [Па];
                             3 - плотность [кг/м^3];
                             4 - местная скорость звука [м/с];
                             5 - динамическая вязкость [Па*с]
                             6 - кинематическая вязкость [м^2*с];
    return: значение выбранного параметра на высоте h
    """
    
    table = pd.read_csv('data_constants/table_atm.csv', names=['h', 'p', 'rho', 'T'], sep=',')

    table_h = table['h']
    table_p = table['p']
    table_T = table['T']
    table_rho = table['rho']

    if parametr == 1:
        return np.interp(h, table_h, table_T)
    elif parametr == 2:
        return np.interp(h, table_h, table_p)
    elif parametr == 3:
        return np.interp(h, table_h, table_rho)
    elif parametr == 4:
        p_h = np.interp(h, table_h, table_p)
        rho_h = np.interp(h, table_h, table_rho)
        k_x = 1.4
        a_h = np.sqrt(k_x * p_h / rho_h)
        return a_h
    elif parametr == 5:
        T_h = np.interp(h, table_h, table_T)
        rho_h = np.interp(h, table_h, table_rho)
        betta_s = 1.458*1e-6
        S = 110.4
        myu = betta_s * T_h**(3/2) / (T_h + S)
        return myu
    elif parametr == 6:
        T_h = np.interp(h, table_h, table_T)
        rho_h = np.interp(h, table_h, table_rho)
        betta_s = 1.458*1e-6
        S = 110.4
        myu = betta_s * T_h**(3/2) / (T_h + S)
        nyu = myu / rho_h
        return nyu
    else:
        print("Ошибка: неверное значение при выборе параметра")

def Cx43(Mah):
    
    """
    Ф-ция закона сопротивления 1943 года
    arguments: число Маха
    return: коэф-т лобового сопротивления Cx
    """
    
    table = pd.read_csv('data_constants/table_cx43.csv', names=['mah', 'cx'], sep=',')
    
    table_mah = table['mah']
    table_cx = table['cx']
    
    return np.interp(Mah, table_mah, table_cx)