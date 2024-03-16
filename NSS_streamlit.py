import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def NSS(m, vi):
    f = vi[0] + vi[1]*((1-np.exp(-m/vi[4]))/(m/vi[4])) \
    + vi[2]*((1-np.exp(-m/vi[4]))/(m/vi[4])-np.exp(-m/vi[4])) \
    + vi[3]*((1-np.exp(-m/vi[5]))/(m/vi[5])-np.exp(-m/vi[5]))
    return f

def list_taux(n, m, vi):
    taux_ = []
    for m in range(n, m+1):
        t = NSS(m, vi)
        taux_.append(t)
    return taux_

def obj(facteur):
    e = (list_taux(1, len(data), facteur)-taux)**2
    scr = np.sum(e)
    return scr

# Streamlit UI
st.title("Nelson Siegel Svensson Model")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
st.caption("Please upload excel file with first column named: Maturité and second named: Taux")

if uploaded_file is not None:
    # Read data from uploaded Excel file
    data = pd.read_excel(uploaded_file)
else:
    # Use default data
    d = {'Maturité': [1,2,3,4,5,8,10,15,20,25,30], 
         'Taux' : [0.01, 0.013, np.nan, np.nan, 0.02, np.nan, 0.024, np.nan, np.nan, 0.03, np.nan]}
    data = pd.DataFrame(data=d)

st.write('---')
maturite = data['Maturité']
taux = data['Taux']

st.subheader('Current Simulated data')
fig, ax = plt.subplots()
ax.scatter(maturite, taux)
ax.set_title('Taux reel (FICTIF)')
ax.set_xlabel('Maturite')
ax.set_ylabel('Taux')
st.pyplot(fig)

transposed_data = data.reset_index(drop=True).T
st.dataframe(transposed_data)

# Definition of parameters
beta0 = np.max(taux)
beta1 = np.max(taux) - np.min(taux)
beta2 = 2*np.mean(taux) - np.max(taux) - np.min(taux)
beta3 = beta2
tau1 = 5
tau2 = 5

vi = [beta0, beta1, beta2, beta3, tau1, tau2]

# Optimization
resultat = minimize(obj, vi, method='Nelder-Mead')
obj_fct = resultat.fun
vo = resultat.x  # Optimal values

# Print different graphs
st.write('---')
col1, col2 = st.columns([1,1])
with col1:
    st.subheader('Taux sans optimisation')
    Taux_initiaux = list_taux(1, len(data), vi)
    data.insert(2, 'Taux initiaux', Taux_initiaux)
    fig, ax = plt.subplots()
    ax.scatter(maturite, taux, label='Taux réel', color='darkblue')
    ax.plot(maturite, Taux_initiaux, label='Taux estimés', color='darkred')
    ax.set_title('Courbe taux réel vs estimé')
    ax.set_xlabel('Maturité')
    ax.set_ylabel('Taux')
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader('Taux avec optimisation')
    taux_estimes = list_taux(1, len(data), vo)
    data.insert(2, 'Taux estimes opt', taux_estimes)
    fig, ax = plt.subplots()
    ax.scatter(maturite, taux, label='Taux réel', color='darkblue')
    ax.plot(maturite, taux_estimes, label='Taux estimés (optimisés)', color='darkred')
    ax.set_title('Courbe taux réel vs estimé (optimisé)')
    ax.set_xlabel('Maturité')
    ax.set_ylabel('Taux')
    ax.legend()
    st.pyplot(fig)
 

# Print optimized parameters as whole numbers
st.subheader('Optimized Parameters')
vi_whole = [round(value, 6) for value in vi]
vo_whole = [round(value, 6) for value in vo]
st.write(f'Based Parameters: **{vi_whole}**')
st.write(f'Optimized Parameters: **{vo_whole}**')
st.caption(f'The parameter were optimised using the Nelder-Mead model in scipy')

st.subheader('Specific Yield per maturities')
maturity = st.number_input('Enter a maturity', min_value=1)
if st.button('Enter'): 
    specific_yield = data[data['Maturité'] == maturity]['Taux estimes opt'].iloc[0]
    #specific_yield = NSS(maturity, vo)
    st.write(f'Yield for {maturity} years : **{specific_yield*100:.4f}%**')