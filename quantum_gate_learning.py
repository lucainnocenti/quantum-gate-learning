{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import theano\n",
    "import theano.tensor as T\n",
    "import theano.tensor.slinalg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def complexrandn(dim1, dim2):\n",
    "    big_matrix = np.random.randn(dim1, dim2, 2)\n",
    "    return big_matrix[:, :, 0] + 1.j * big_matrix[:, :, 1]\n",
    "\n",
    "def complex2bigreal(matrix):\n",
    "    row1 = np.concatenate((np.real(matrix), -np.imag(matrix)), axis=1)\n",
    "    row2 = np.concatenate((np.imag(matrix), np.real(matrix)), axis=1)\n",
    "    return np.concatenate((row1, row2), axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 2.26167452,  1.4940247 ,  7.23111148, -1.12055925],\n",
       "       [ 2.0705593 ,  0.35627242,  2.5004635 , -0.77347502],\n",
       "       [-7.23111148,  1.12055925,  2.26167452,  1.4940247 ],\n",
       "       [-2.5004635 ,  0.77347502,  2.0705593 ,  0.35627242]])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "H = complexrandn(2, 2)\n",
    "# print(H)\n",
    "Hp1 = np.concatenate((-np.imag(H), -np.real(H)), axis=1)\n",
    "Hp2 = np.concatenate((np.real(H), -np.imag(H)), axis=1)\n",
    "Hp = np.concatenate((Hp1, Hp2), axis=0)\n",
    "x = T.dscalar('x')\n",
    "expH = T.slinalg.expm(x * Hp)\n",
    "# theano.pp(expH)\n",
    "expH_flat = T.flatten(expH)\n",
    "def fn(i, M, x):\n",
    "    return T.grad(M[i], x)\n",
    "J, updates = theano.scan(fn=fn,\n",
    "                         sequences=[T.arange(expH_flat.shape[0])],\n",
    "                         non_sequences=[expH_flat, x])\n",
    "gexpH = J.reshape(expH.shape)\n",
    "f = theano.function([x], gexpH)\n",
    "f(2.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(16, 8, 8)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],\n",
       "       [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],\n",
       "       [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],\n",
       "       [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],\n",
       "       [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],\n",
       "       [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],\n",
       "       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],\n",
       "       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import qutip\n",
    "import itertools\n",
    "from collections import OrderedDict\n",
    "\n",
    "# get_sigmas_index gets a tuple as input and gives back\n",
    "# a length-16 array of zeros with only one element equal to 1\n",
    "def get_sigmas_index(indices):\n",
    "    all_zeros = np.zeros(4 * 4)\n",
    "    all_zeros[indices[0] * 4 + indices[1]] = 1.\n",
    "    return all_zeros\n",
    "\n",
    "# generate all tensor products of sigmas\n",
    "sigmas = [qutip.qeye(2), qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]\n",
    "sigma_pairs = []\n",
    "for idx1 in range(4):\n",
    "    for idx2 in range(4):\n",
    "        sigma_pairs.append(\n",
    "            complex2bigreal(\n",
    "                1j * qutip.tensor(sigmas[idx1], sigmas[idx2]).data.toarray()))\n",
    "sigma_pairs = np.asarray(sigma_pairs)\n",
    "\n",
    "print(sigma_pairs.shape)\n",
    "\n",
    "# J is the theano vector containing all the interactions strengths\n",
    "J = T.dvector('J')\n",
    "H = T.tensordot(J, sigma_pairs, axes=1)\n",
    "\n",
    "expH = T.slinalg.expm(H)\n",
    "\n",
    "f = theano.function([J], H)\n",
    "f(get_sigmas_index((0, 1)))\n",
    "# theano.printing.pydotprint(H, 'testPNG.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.all(np.asarray((0, 1, 3, 5)) < 6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def chars2pair(chars):\n",
    "    out_pair = []\n",
    "    for idx in range(len(chars)):\n",
    "        if chars[idx] == 'x':\n",
    "            out_pair.append(1)\n",
    "        elif chars[idx] == 'y':\n",
    "            out_pair.append(2)\n",
    "        elif chars[idx] == 'z':\n",
    "            out_pair.append(3)\n",
    "        else:\n",
    "            raise ValueError('chars must contain 2 characters, each of'\n",
    "                             'which equal to either x, y, or z')\n",
    "    return tuple(out_pair)\n",
    "\n",
    "class QubitNetwork:\n",
    "    def __init__(self, num_qubits,\n",
    "                 interactions='all', self_interactions='all',\n",
    "                 system_qubits=None):\n",
    "        self.num_qubits = num_qubits\n",
    "        # we store all the possible pairs for convenience\n",
    "        self.pairs = list(itertools.combinations(range(self.num_qubits), 2))\n",
    "        # decode_interactions_dict fills the self.active_Js variable\n",
    "        self.active_Js = self.decode_interactions(interactions)\n",
    "        self.active_hs = self.decode_self_interactions(self_interactions)\n",
    "        self.num_interactions = self.count_interactions()\n",
    "        self.num_self_interactions = self.count_self_interactions()\n",
    "        self.Js_factors, self.hs_factors = self.build_H_components()\n",
    "        self.initial_state = self.build_initial_state_vector()\n",
    "        \n",
    "        # Define which qubits belong to the system. The others are all\n",
    "        # assumed to be ancilla qubits\n",
    "        if system_qubits is None:\n",
    "            self.system_qubits = tuple(range(num_qubits // 2))\n",
    "        elif np.all(np.asarray(system_qubits) < num_qubits):\n",
    "            self.system_qubits = tuple(system_qubits)\n",
    "        else:\n",
    "            raise ValueError('Invalid value for system_qubits.')\n",
    "\n",
    "    def decode_interactions(self, interactions):\n",
    "        if interactions == 'all':\n",
    "            allsigmas = [item[0] + item[1]\n",
    "                         for item in itertools.product(['x', 'y', 'z'], repeat=2)]\n",
    "            return OrderedDict([(pair, allsigmas) for pair in self.pairs])\n",
    "        elif isinstance(interactions, tuple):\n",
    "            if interactions[0] == 'all':\n",
    "                d = {pair: interactions[1] for pair in self.pairs}\n",
    "                return OrderedDict(d)\n",
    "        elif (isinstance(interactions, dict) and\n",
    "              all(isinstance(k, tuple) for k in interactions.keys())):\n",
    "            return OrderedDict(interactions)\n",
    "        else:\n",
    "            raise ValueError('Invalid value given for interactions.')\n",
    "    \n",
    "    def decode_self_interactions(self, self_interactions):\n",
    "        if self_interactions == 'all':\n",
    "            return OrderedDict(\n",
    "                {idx: ['x', 'y', 'z'] for idx in range(self.num_qubits)})\n",
    "        elif isinstance(self_interactions, tuple):\n",
    "            if self_interactions[0] == 'all':\n",
    "                d = {idx: self_interactions[1] for idx in range(self.num_qubits)}\n",
    "                return OrderedDict(d)\n",
    "            else:\n",
    "                raise ValueError('Invalid value for self_interactions.')\n",
    "        else:\n",
    "            raise ValueError('Invalid value of self_interactions.')\n",
    "            \n",
    "    \n",
    "    def count_interactions(self):\n",
    "        count = 0\n",
    "        for k, v in self.active_Js.items():\n",
    "            count += len(v)\n",
    "        return count\n",
    "\n",
    "    def count_self_interactions(self):\n",
    "        count = 0\n",
    "        for k, v in self.active_hs.items():\n",
    "            count += len(v)\n",
    "        return count\n",
    "    \n",
    "    def build_H_components(self):\n",
    "        terms_template = [qutip.qeye(2) for _ in range(self.num_qubits)]\n",
    "        Js_factors = []\n",
    "        hs_factors = []\n",
    "        for pair, directions in self.active_Js.items():\n",
    "            # - directions is a list of elements like ss below\n",
    "            # - ss is a two-character string specifying an interaction\n",
    "            # direction, e.g. 'xx' or 'xy' or 'zy'\n",
    "            for ss in directions:\n",
    "                term = terms_template\n",
    "                term[pair[0]] = sigmas[chars2pair(ss)[0]]\n",
    "                term[pair[1]] = sigmas[chars2pair(ss)[1]]\n",
    "                term = complex2bigreal(1j * qutip.tensor(term).data.toarray())\n",
    "                Js_factors.append(term)\n",
    "\n",
    "        for qubit_idx, direction in self.active_hs.items():\n",
    "            # - now direction is a list of characters among 'x', 'y' and 'z',\n",
    "            # - s is either 'x', 'y', or 'z'\n",
    "            for s in direction:\n",
    "                term = terms_template\n",
    "                term[qubit_idx] = sigmas[chars2pair(s)[0]]\n",
    "                term = complex2bigreal(1j * qutip.tensor(term).data.toarray())\n",
    "                hs_factors.append(term)\n",
    "\n",
    "        return np.asarray(Js_factors), np.asarray(hs_factors)\n",
    "    \n",
    "    def build_initial_state_vector(self):\n",
    "        state = qutip.tensor([qutip.basis(2, 0) for _ in range(self.num_qubits)])\n",
    "        state = state.data.toarray()\n",
    "        state = np.concatenate((np.real(state), np.imag(state)), axis=0)\n",
    "        return state"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0, 1, 2)"
      ]
     },
     "execution_count": 134,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "net = QubitNetwork(4, interactions=('all', ['zz']),\n",
    "                   self_interactions=('all', 'x'),\n",
    "                   system_qubits=[0, 1, 2])\n",
    "net.system_qubits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/latex": [
       "Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isherm = True\\begin{equation*}\\left(\\begin{array}{*{11}c}1.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0\\\\\\end{array}\\right)\\end{equation*}"
      ],
      "text/plain": [
       "Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isherm = True\n",
       "Qobj data =\n",
       "[[ 1.  0.  0.  0.]\n",
       " [ 0.  0.  0.  0.]\n",
       " [ 0.  0.  0.  0.]\n",
       " [ 0.  0.  0.  0.]]"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "foo = qutip.ket2dm(qutip.Qobj(net.initial_state))\n",
    "foo.dims = [[2, 2, 2, 2, 2] for _ in range(2)]\n",
    "foo.ptrace([0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/latex": [
       "Quantum object: dims = [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3]], shape = [243, 243], type = oper, isherm = True\\begin{equation*}\\left(\\begin{array}{*{11}c}1.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\\\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0 & 0.0 & \\cdots & 0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\\\\end{array}\\right)\\end{equation*}"
      ],
      "text/plain": [
       "Quantum object: dims = [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3]], shape = [243, 243], type = oper, isherm = True\n",
       "Qobj data =\n",
       "[[ 1.  0.  0. ...,  0.  0.  0.]\n",
       " [ 0.  0.  0. ...,  0.  0.  0.]\n",
       " [ 0.  0.  0. ...,  0.  0.  0.]\n",
       " ..., \n",
       " [ 0.  0.  0. ...,  0.  0.  0.]\n",
       " [ 0.  0.  0. ...,  0.  0.  0.]\n",
       " [ 0.  0.  0. ...,  0.  0.  0.]]"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "qutip.ket2dm(qutip.tensor([qutip.basis(3, 0) for _ in range(5)]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10"
      ]
     },
     "execution_count": 131,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "net.num_interactions + net.num_self_interactions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.00328995, -0.00179964,  0.00137534, ...,  0.00439022,\n",
       "        -0.01401882, -0.01618361],\n",
       "       [-0.00179964,  0.00098442, -0.00075233, ..., -0.0024015 ,\n",
       "         0.00766845,  0.00885262],\n",
       "       [ 0.00137534, -0.00075233,  0.00057495, ...,  0.0018353 ,\n",
       "        -0.00586048, -0.00676546],\n",
       "       ..., \n",
       "       [ 0.00439022, -0.0024015 ,  0.0018353 , ...,  0.00585845,\n",
       "        -0.01870715, -0.02159592],\n",
       "       [-0.01401882,  0.00766845, -0.00586048, ..., -0.01870715,\n",
       "         0.05973557,  0.06895997],\n",
       "       [-0.01618361,  0.00885262, -0.00676546, ..., -0.02159592,\n",
       "         0.06895997,  0.0796088 ]])"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "H_factors = np.concatenate((net.hs_factors, net.Js_factors), axis=0)\n",
    "# J is the theano vector containing all the interactions strengths\n",
    "J = T.dvector('J')\n",
    "H = T.tensordot(J, H_factors, axes=1)\n",
    "\n",
    "expH = T.slinalg.expm(H)\n",
    "# initial_dm = net.initial_state * net.initial_state.T\n",
    "expH_times_state = T.dot(expH, net.initial_state)\n",
    "dm = expH_times_state * expH_times_state.T\n",
    "\n",
    "# expH_flat = T.flatten(expH)\n",
    "# def fn(i, matrix, x):\n",
    "#     return T.grad(matrix[i], x)\n",
    "# expH_flat_grad, updates = theano.scan(fn=fn,\n",
    "#                                       sequences=T.arange(expH_flat.shape[0]),\n",
    "#                                       non_sequences=[expH_flat, J])\n",
    "\n",
    "# grads = theano.function([J], expH_flat_grad.reshape(expH.shape))\n",
    "\n",
    "initial_parameters = np.random.randn(net.num_interactions + net.num_self_interactions)\n",
    "f = theano.function([J], dm)\n",
    "f(initial_parameters)"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [conda env:theano]",
   "language": "python",
   "name": "conda-env-theano-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
