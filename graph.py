import numpy as np
import tensorflow as tf
import itertools as itr

class Data_Node():
    def __init__(self):
        self.pixel_batch = tf.placeholder(tf.complex128, shape = (None, None))
        conj_pixel = tf.conj(self.pixel_batch)
        self.output = tf.einsum('ab, ac -> abc', self.pixel_batch, conj_pixel)

class Op_Node():
    def __init__(self, input_nodes, shape):
        self.input_nodes = input_nodes
        self.shape = shape
        self.size = shape[0] * shape[1] * shape[2]

    def create_node_ops(self, index, param_var):
        self.index = index
        self.param_var = param_var
        with tf.variable_scope(str(index)):
            self.create_params()
            self.create_hermitian_ops()
            self.create_unitary_ops()
            self.create_contract_ops()

    def create_params(self):
        num_diag = self.size
        num_off_diag = int(0.5 * self.size * (self.size - 1))
        total_params = num_diag + 2 * num_off_diag
        start_slice = self.index * total_params
        diag_end = start_slice + num_diag
        real_end = diag_end + num_off_diag
        imag_end = real_end + num_off_diag
        self.diag_params = tf.slice(self.param_var, [start_slice], [num_diag])
        self.real_off_params = tf.slice(self.param_var, [diag_end], [num_off_diag])
        self.imag_off_params = tf.slice(self.param_var, [real_end], [num_off_diag])

    def create_hermitian_ops(self):
        herm_shape = (self.size, self.size)
        diag_part = tf.diag(self.diag_params)
        off_diag_indices = [(i, j) for i in range(self.size) for j in range(i + 1, self.size)]
        real_off_diag_part = tf.scatter_nd(
            indices = off_diag_indices,
            updates = self.real_off_params,
            shape = herm_shape)
        imag_off_diag_part = tf.scatter_nd(
            indices = off_diag_indices,
            updates = self.imag_off_params,
            shape = herm_shape)
        imag_whole = imag_off_diag_part - tf.transpose(imag_off_diag_part)
        real_whole = diag_part + real_off_diag_part + tf.transpose(real_off_diag_part)
        self.herm_matrix = tf.complex(real_whole, imag_whole)

    def create_unitary_ops(self):
        (eigenvalues, eigenvectors) = tf.linalg.eigh(self.herm_matrix)
        eig_exponential = tf.exp(1j * eigenvalues)
        diag_exponential = tf.diag(eig_exponential)
        unitary_matrix = tf.einsum(
            'bc,cd,ed->be',
            eigenvectors,
            diag_exponential,
            tf.conj(eigenvectors))
        self.unitary_tensor = tf.reshape(unitary_matrix, self.shape)

    def create_contract_ops(self):
        (left_node, right_node) = self.input_nodes
        left_input = left_node.output
        right_input = right_node.output
        ancilla = tf.one_hot(
            indices = 0,
            depth = self.shape[2],
            dtype = tf.complex128)
        uni_and_anc = tf.einsum('bcdef,d->bcef', self.unitary_tensor, ancilla)
        contract_left = tf.einsum('bcde,abf->afcde', uni_and_anc, left_input)
        full_contract = tf.einsum('abcde,acf->abfde', contract_left, right_input)
        self.output = tf.einsum('abcde,bcfe->adf', full_contract, tf.conj(uni_and_anc))

class Graph():
    def __init__(self, num_pixels, op_shape):
        self.op_shape = op_shape
        self.num_op_params = (op_shape[0] * op_shape[1] * op_shape[2]) ** 2
        self.num_pixels = num_pixels
        self.generate_graph()

    def generate_graph(self):
        self.data_nodes = create_data_layer(self.num_pixels)
        self.op_nodes = create_all_op_nodes(self.num_pixels, self.data_nodes, self.op_shape)
        self.create_params_var()
        num_layers = len(self.op_nodes.keys())
        root_node = self.op_nodes[num_layers - 1][0]
        self.probabilities = tf.real(tf.matrix_diag_part(root_node.output))
        self.init = tf.global_variables_initializer()

    def create_params_var(self):
        op_layers = self.op_nodes.values()
        self.all_ops = list(itr.chain.from_iterable(op_layers))
        self.param_var = tf.get_variable(
            'param_var',
            shape = [self.num_op_params * len(self.all_ops)],
            dtype = tf.float64,
            initializer = tf.random_normal_initializer)
        self.update = tf.placeholder(tf.float64)
        self.update_params = tf.assign_add(self.param_var, self.update)
        for (index, op) in enumerate(self.all_ops):
            op.create_node_ops(index, self.param_var)

    def run_graph(self, sess, image_batch):
        #fd_dict = {node.pixel_batch : image_batch[:, pixel, :] for (pixel, node) in enumerate(self.data_nodes)}
        fd_dict = self.create_pixel_dict(image_batch)
        probabilities = sess.run(self.probabilities, feed_dict = fd_dict)
        return probabilities

    def create_pixel_dict(self, image_batch):
        pixel_dict = {}
        for (index, node) in enumerate(self.data_nodes):
            quad = index // 16
            quad_quad = (index % 16) // 4
            pos = index % 4
            row = (pos // 2) + 2*(quad_quad // 2) + 4*(quad // 2)
            col = (pos % 2) + 2*(quad_quad % 2) + 4*(quad % 2)
            pixel = col + 8 * row
            pixel_dict[node.pixel_batch] = image_batch[:, pixel, :]
        return pixel_dict

def create_data_layer(num_pixels):
    data_nodes = []
    for pixel in range(num_pixels):
        node = Data_Node()
        data_nodes.append(node)
    return data_nodes

def create_all_op_nodes(num_pixels, data_layer, shape):
    num_layers = int(np.log2(num_pixels))
    op_nodes = {}
    op_nodes[0] = create_op_layer(data_layer, shape)
    for layer in range(1, num_layers):
        prev_layer = op_nodes[layer - 1]
        op_nodes[layer] = create_op_layer(prev_layer, shape)
    return op_nodes

def create_op_layer(prev_layer, shape):
    op_layer = []
    for i in range(0, len(prev_layer), 2):
        input_nodes = (prev_layer[i], prev_layer[i + 1])
        op_node = Op_Node(input_nodes, shape)
        op_layer.append(op_node)
    return op_layer
