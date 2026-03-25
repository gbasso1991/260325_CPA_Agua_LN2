#%%%
import os
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura 
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
    
    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return timestamp,time,temp_CH1, temp_CH2
#%% Transicion de fase
def detectar_TF_y_plot(t,T,T_central=0,delta_T=0.2,umbral_dTdt=0.15,min_puntos=5,plot=True,identif=None):
    """
    Detecta mesetas de transición de fase en una curva Temperatura vs Tiempo
    y opcionalmente genera un gráfico con la región identificada.

    La meseta se define como una región donde:
    - La temperatura se mantiene dentro de un intervalo alrededor de T_central
    - La derivada temporal |dT/dt| es menor que un umbral dado
    - Los puntos cumplen continuidad temporal (segmentos consecutivos)
    - La longitud del segmento supera un mínimo de puntos (min_puntos)

    Parámetros
    ----------
    t : array_like
        Tiempo [s]
    T : array_like
        Temperatura [°C]
    T_central : float, opcional
        Temperatura central de la transición (default: 0 °C)
    delta_T : float, opcional
        Tolerancia en temperatura (± delta_T) (default: 0.2 °C)
    umbral_dTdt : float, opcional
        Umbral máximo para |dT/dt| [°C/s] (default: 0.15 °C/s)
    min_puntos : int, opcional
        Número mínimo de puntos consecutivos para validar una meseta (default: 5)
    plot : bool, opcional
        Si True, genera la figura con los resultados (default: True)

    Retorna
    -------
    mesetas : list of dict
        Lista de mesetas detectadas. Cada elemento contiene:
        - "t_inicio" : tiempo inicial [s]
        - "t_fin"    : tiempo final [s]
        - "duracion" : duración de la meseta [s]
        - "T_media"  : temperatura media en la meseta [°C]

    fig : matplotlib.figure.Figure o None
        Figura generada (si plot=True)
    ax : matplotlib.axes.Axes o None
        Eje de temperatura
    ax2 : matplotlib.axes.Axes o None
        Eje de derivada dT/dt

    Notas
    -----
    - La derivada dT/dt se calcula mediante diferencias finitas (np.gradient).
    - La segmentación en bloques continuos evita identificar puntos aislados
      (ruido) como mesetas físicas.
    - El método es especialmente útil en experimentos térmicos donde la
      transición de fase se manifiesta como una meseta (ej: fusión del agua).
    - Para datos ruidosos se recomienda suavizar previamente la señal de temperatura."""
    dT_dt = np.gradient(T, t) # --- Derivada ---


    mask = ((T > (T_central - delta_T)) & (T < (T_central + delta_T)) & (np.abs(dT_dt) < umbral_dTdt))     # --- Filtro ---

    idx = np.where(mask)[0]

    if len(idx) == 0:
        return [], None, None, None

    segmentos = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)     # --- Segmentos continuos ---

    mesetas = []
    for seg in segmentos:
        if len(seg) >= min_puntos:
            t_ini = t[seg[0]]
            t_fin = t[seg[-1]]

            mesetas.append({"t_inicio": t_ini,"t_fin": t_fin,
                "duracion": t_fin - t_ini,"T_media": np.mean(T[seg])})

    if plot:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10,7),sharex=True, constrained_layout=True)

        ax.plot(t, T, '.-', label='Temperatura')
        ax2.plot(t, dT_dt, '.-', label='dT/dt')

        # Umbrales derivada
        ax2.axhline(umbral_dTdt, color='k', ls='--')
        ax2.axhline(-umbral_dTdt, color='k', ls='--')

        # --- Mesetas ---
        for i, m in enumerate(mesetas):
            mask_m = (t >= m["t_inicio"]) & (t <= m["t_fin"])

            label = f'T. Fase ({m["duracion"]:.1f} s)' if i == 0 else None

            # Curva resaltada
            ax.plot(t[mask_m], T[mask_m], 'g-', lw=3, label=label)

            # Sombreado en ambos plots
            ax.axvspan(m["t_inicio"], m["t_fin"], color='g', alpha=0.2)
            ax2.axvspan(m["t_inicio"], m["t_fin"], color='g', alpha=0.2)

        # --- Labels ---
        ax.set_ylabel('T (°C)')
        ax2.set_ylabel('dT/dt (°C/s)')
        ax2.set_xlabel('t (s)')
        ax.set_title(identif+'\nTransición de fase S-L')

        for a in (ax, ax2):
            a.grid()
            a.legend()

        # --- Inset (primera meseta) ---
        if mesetas:
            m = mesetas[0]
            mask_m = (t >= m["t_inicio"]) & (t <= m["t_fin"])

            axin = ax.inset_axes([0.5, 0.1, 0.45, 0.45])
            axin.plot(t, T, 'k-')
            axin.plot(t[mask_m], T[mask_m], 'g-', lw=2)

            axin.axhline(T_central - delta_T, ls='--', color='k')
            axin.axhline(T_central + delta_T, ls='--', color='k')

            axin.set_xlim(m["t_inicio"] - 5, m["t_fin"] + 5)
            axin.set_ylim(T_central - 2*delta_T, T_central + 2*delta_T)

            axin.grid()
            ax.indicate_inset_zoom(axin)

        return mesetas, fig, ax, ax2

    return mesetas, None, None, None
#%% 1,2,3 CPA100/050/000 - 500 uL - BT1 
path_1 = '260325_114944_CPA100_BT1_500uL.csv'
path_2 = '260325_120927_CPA050_BT1_500uL.csv'
path_3 = '260325_120137_agua_BT1_500uL.csv'

_,t_100,T_100,_ = lector_templog(path_1)
_,t_050,T_050,_ = lector_templog(path_2)
_,t_agua,T_agua,_ = lector_templog(path_3)


fig100, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 100% 50% y 0%  - enfriado en LN2 - expuesto a BT1',loc='left')
ax.plot(t_100,T_100,'.-',label='CPA100',alpha=0.8)
ax.plot(t_050,T_050,'.-',label='CPA050',alpha=0.8)
ax.plot(t_agua,T_agua,'.-',label='Agua',alpha=0.8)

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)
plt.show()
#%% 4,5,6 CPA100/050/000 - 500 uL - BT1 Vapor  
path_4 = '260325_121904_CPA100_BT1_500uL_vapor.csv'
path_5 = '260325_123339_CPA050_BT1_500uL_vapor.csv'
path_6 = '260325_124612_agua_BT1_500uL_vapor.csv'

_,t_100,T_100,_ = lector_templog(path_4)
_,t_050,T_050,_ = lector_templog(path_5)
_,t_agua,T_agua,_ = lector_templog(path_6)

#%
fig100, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 100% 50% y 0%  - enfriado en vapor LN2 - expuesto a BT1',loc='left')
ax.plot(t_100,T_100,'.-',label='CPA100',alpha=0.8)
#ax.plot(t_050,T_050,'.-',label='CPA050',alpha=0.8)
ax.plot(t_agua,T_agua,'.-',label='Agua',alpha=0.8)

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)

plt.show()
