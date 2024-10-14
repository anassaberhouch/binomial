import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st


class BlackScholesModel:
    def __init__(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility):
        self.S0 = spot_price  # Current stock price
        self.K = strike_price  # Strike price
        self.T = time_to_maturity  # Time to maturity (in years)
        self.r = risk_free_rate  # Risk-free interest rate
        self.sigma = volatility  # Volatility of the underlying asset

    def calculate_option_price(self, option_type='call'):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if option_type == 'call':
            option_price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'put':
            option_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")

        return option_price
class binomialmodel :
    def __init__(self,r,T,K,S,sigma,option_type,n):
        self.T=T
        self.K = K
        self.S = S
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.n = n
    
    def binomial_pricing(self):
        dt = self.T / self.n
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        option_tree = np.zeros((self.n+1, self.n+1))
        stock_tree = np.zeros((self.n+1, self.n+1))
        
        for j in range(self.n+1):
            stock_tree[self.n, j] = self.S * (u ** (self.n - j)) * (d ** j)
            option_tree[self.n, j] = max(0, self.option_type * (stock_tree[self.n, j] - self.K))
        
        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S * (u ** (i - j)) * (d ** j)
                option_tree[i, j] = max(0, np.exp(-self.r * dt) * (p * option_tree[i+1, j] + (1-p) * option_tree[i+1, j+1]))
        
        return option_tree[0, 0],stock_tree,option_tree
class trinomialmodel:
    def __init__(self, r, T, K, S, sigma, option_type, n):
        self.T = T
        self.K = K
        self.S = S
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.n = n

    def trinomial_pricing(self):
        dt = self.T / self.n
        u = np.exp(self.sigma * np.sqrt(2 * dt))  # Up factor
        d = 1 / u  # Down factor
        m = 1  # Middle stays the same

        # Risk-neutral probabilities
        pu = 0.25 + ((self.r - 0.5 * self.sigma ** 2) * np.sqrt(dt)) / (2 * self.sigma)
        pd = 0.25 - ((self.r - 0.5 * self.sigma ** 2) * np.sqrt(dt)) / (2 * self.sigma)
        pm = 1 - pu - pd  # Middle probability

        # Option tree and stock tree initialization
        option_tree = np.zeros((self.n + 1, 2 * self.n + 1))
        stock_tree = np.zeros((self.n + 1, 2 * self.n + 1))

        # Stock prices at maturity
        for j in range(2 * self.n + 1):
            stock_tree[self.n, j] = self.S * (u ** (self.n - j)) * (d ** j)
            option_tree[self.n, j] = max(0, self.option_type * (stock_tree[self.n, j] - self.K))

        # Step back through the tree
        for i in range(self.n - 1, -1, -1):
            for j in range(2 * i + 1):
                stock_tree[i, j] = self.S * (u ** (i - j)) * (d ** j)
                option_tree[i, j] = np.exp(-self.r * dt) * (
                    pu * option_tree[i + 1, j + 2] +
                    pm * option_tree[i + 1, j + 1] +
                    pd * option_tree[i + 1, j]
                )

        return option_tree[0, 0], stock_tree, option_tree

class BinomialTree:
    def __init__(self, matrix):
        self.G = nx.DiGraph()
        self.matrix = matrix
        self.labels = {}
        self.pos = None
        self.create_tree()
        self.set_node_labels()
    def create_tree(self):
        rows = len(self.matrix)
        for i in range(rows):
            for j in range(i+1):
                parent = (i, j)
                if i + 1 < rows:
                    left_child = (i+1, j+1)
                    right_child = (i+1, j)
                    self.G.add_edge(parent, right_child)
                    self.G.add_edge(parent, left_child)

    def set_node_labels(self):
        for node in self.G.nodes():
            i, j = node
            try:
                self.labels[node] ="$S_{"+str(i)+"}"+f"$={self.matrix[i][j]:.2f}"
            except Exception:
                self.labels[node] ='outofbound'

    def draw_tree(self):
        if self.pos is None:
            self.pos = nx.nx_pydot.graphviz_layout(self.G, prog='dot')

        scale = 1.0 / len(self.matrix)
        pos = nx.nx_pydot.graphviz_layout(self.G, prog='dot')
        # pos_graphviz = {k: (-v[1] * scale, v[0] * scale) for k, v in pos.items()}

        pos_graphviz = {k: (-v[1], v[0]) for k, v in self.pos.items()}

        fig, ax = plt.subplots(figsize=(10, int(1/scale)))

        nx.draw(self.G, pos_graphviz, with_labels=True, labels=self.labels, node_size=1000, node_color='skyblue', font_size=10, font_color='black')
        plt.show()
class TrinomialTree:
    def __init__(self, matrix):
        self.G = nx.DiGraph()
        self.matrix = matrix
        self.labels = {}
        self.pos = None
        self.create_tree()
        self.set_node_labels()
    def create_tree(self):
        rows = len(self.matrix)
        for i in range(rows):
            for j in range(2*i+1):
                parent = (i, j)
                if i+1  < rows:
                    left_child = (i+1, j+2)
                    middle_child = (i+1, j+1)
                    right_child = (i+1, j)
                    self.G.add_edge(parent, right_child)
                    self.G.add_edge(parent, middle_child)
                    self.G.add_edge(parent, left_child)

    def set_node_labels(self):
        for node in self.G.nodes():
            i, j = node
            try:
                self.labels[node] ="$S_{"+str(i)+"}"+f"$={self.matrix[i][j]:.2f}"
            except Exception:
                self.labels[node] ='outofbound'

    def draw_tree(self):
        if self.pos is None:
            self.pos = nx.nx_pydot.graphviz_layout(self.G, prog='dot')

        pos_graphviz = {k: (-v[1], v[0]) for k, v in self.pos.items()}

        fig, ax = plt.subplots(figsize=(10, 5))

        nx.draw(self.G, pos_graphviz, with_labels=True, labels=self.labels, node_size=1000, node_color='skyblue', font_size=10, font_color='black')
        plt.show()


import streamlit as st

from enum import Enum
from datetime import datetime, timedelta

# Third party imports
import streamlit as st

st.title("Option Pricing App")

# Sidebar for user input
st.sidebar.header("Option Parameters")
option_type = st.sidebar.selectbox("Option Type", ['Call', 'Put'])
S0 = st.sidebar.number_input("Spot Price (S0)", min_value=0.01, value=100.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0)
T = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05)
sigma = st.sidebar.number_input("Volatility (sigma)", min_value=0.0, value=0.2)

# Model selection
model_choice = st.sidebar.radio("Select Pricing Model", ["Black-Scholes", "Binomial", "Trinomial"])

if model_choice == "Black-Scholes":
    model = BlackScholesModel(S0, K, T, r, sigma)
    option_price = model.calculate_option_price(option_type.lower())
elif model_choice == "Binomial":
    n = st.sidebar.number_input("Number of Steps (n)", min_value=1, step=1, value=3)
    model = binomialmodel(r, T, K, S0, sigma, 1 if option_type.lower() == "call" else -1, n)
    option_price, _, _ = model.binomial_pricing()
elif model_choice == "Trinomial":
    n = st.sidebar.number_input("Number of Steps (n)", min_value=1, step=1, value=3)
    model = trinomialmodel(r, T, K, S0, sigma, 1 if option_type.lower() == "call" else -1, n)
    option_price, _, _ = model.trinomial_pricing()

# Display the option price
st.subheader("Option Price")
st.write(f"The {option_type} option price is: {option_price:.2f}")

# Optional: Display binomial or trinomial tree
if model_choice == "Binomial":
    st.subheader("Binomial Option Pricing Tree")
    st.write("Note: Displaying the binomial tree for educational purposes. It can be large for a large number of steps.")
    
    # Create binomial tree and draw it
    if st.sidebar.button('Draw Binomial Tree'):
        option_price, stock_tree, option_tree = model.binomial_pricing()
        bin_tree = BinomialTree(option_tree)
        bin_tree.draw_tree()
        st.pyplot()

elif model_choice == "Trinomial":
    st.subheader("Trinomial Option Pricing Tree")
    st.write("Note: Displaying the trinomial tree for educational purposes. It can be large for a large number of steps.")
    
    # Create trinomial tree and draw it
    if st.sidebar.button('Draw Trinomial Tree'):
        option_price, stock_tree, option_tree = model.trinomial_pricing()
        tri_tree = TrinomialTree(option_tree)
        tri_tree.draw_tree()
        st.pyplot()