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

def list_taux(data, vi):
    taux_ = []
    for m in data['Maturité']:
        t = NSS(m, vi)
        taux_.append(t)
    return taux_

def obj(facteur):
    e = (list_taux(data, facteur)-data['Taux'])**2
    scr = np.sum(e)
    return scr

def fill_missing_maturities(data):
    min_maturity = data['Maturité'].min()
    max_maturity = data['Maturité'].max()
    all_maturities = np.arange(min_maturity, max_maturity + 1)
    missing_maturities = set(all_maturities) - set(data['Maturité'])
    missing_data = pd.DataFrame({'Maturité': list(missing_maturities), 'Taux': np.nan})
    data = pd.concat([data, missing_data], ignore_index=True)
    data.sort_values('Maturité', inplace=True)
    return data

def download_excel():
    transposed_data.to_excel("selected_data.xlsx", index=False)
    with open("selected_data.xlsx", "rb") as file:
        btn = file.read()
    return btn

# Streamlit UI
st.image("heconomie.jpg")
st.title("Nelson Siegel Svensson Model")
st.write("Made by the Fixed Income Team of HEConomie")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
st.caption("Please upload excel file with first column named: Maturité and second named: Taux")

# Frequency selection
frequency = st.selectbox("Select Frequency", ["Yearly", "Semiannual", "Quarterly", "Monthly"])
if frequency == "Yearly":
    freq_factor = 1
elif frequency == "Semiannual":
    freq_factor = 2
elif frequency == "Quarterly":
    freq_factor = 4
elif frequency == "Monthly":
    freq_factor = 12

if uploaded_file is not None:
    # Read data from uploaded Excel file
    data = pd.read_excel(uploaded_file)
    if freq_factor != 1:
        # Interpolate data to match the selected frequency
        new_index = np.arange(data['Maturité'].min(), data['Maturité'].max() + 1, 1 / freq_factor)
        data = data.set_index('Maturité').reindex(new_index).reset_index()
        data['Maturité'] = data.index + 1
    data = fill_missing_maturities(data)
    data.reset_index(drop=True, inplace=True)
else:
    # Use default data with maturities from 1 to 30 years
    maturities = np.arange(1, 31 * freq_factor + 1)
    # Simulating Taux with square root function
    random_taux = np.sqrt(maturities) * np.random.uniform(0.01, 0.05, size=len(maturities))
    d = {'Maturité': maturities, 'Taux': random_taux}
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
    Taux_initiaux = list_taux(data, vi)
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
    taux_estimes = list_taux(data, vo)
    data.insert(2, 'Taux estimes opt', taux_estimes)
    fig, ax = plt.subplots()
    ax.scatter(maturite, taux, label='Taux réel', color='darkblue')
    ax.plot(maturite, taux_estimes, label='Taux estimés (optimisés)', color='darkred')
    ax.set_title('Courbe taux réel vs estimé (optimisé)')
    ax.set_xlabel('Maturité')
    ax.set_ylabel('Taux')
    ax.legend()
    st.pyplot(fig)

selected_data = data[['Maturité', 'Taux estimes opt']]
transposed_data = selected_data.T
st.dataframe(transposed_data)
st.download_button(label="Download data as Excel", data=download_excel(), file_name="selected_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
 

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
    specific_yield_NSS = NSS(maturity, vo)
    formatted_yield = "{:.3f}%".format(specific_yield_NSS * 100)
    st.write(f"The yield for {maturity} ({frequency}) : **{formatted_yield}**")

