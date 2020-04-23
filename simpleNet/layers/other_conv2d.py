import numpy as np
from simpleNet.layers.Layer import Layer


class Conv2d_other(Layer):
    def __init__(self, in_channel:int, out_channel:int, kernel_size, stride:int=1, padding = 0):
        super().__init__()
        self.name = "Conv2D"

        self.param = {"pad":padding, "stride":stride}
        w = np.random.rand(out_channel, in_channel, kernel_size, kernel_size).astype(np.float64)
        b = np.random.rand(out_channel).astype(np.float64)
        self.weights = {"w": w, "b": b}

    def __call__(self, x):
        z, cache = conv_forward_naive(x, self.weights["w"], self.weights["b"], self.param)
        self.cache = cache
        return z

    def backwards(self, da):
        dx, dw, db = conv_backward_naive(da, self.cache)
        self.cached_grad = {"w":dw, "b":db}
        return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad_num = conv_param['pad']
  stride = conv_param['stride']
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_prime = (H+2*pad_num-HH) // stride + 1
  W_prime = (W+2*pad_num-WW) // stride + 1
  out = np.zeros([N,F,H_prime,W_prime])
  #im2col
  for im_num in range(N):
      im = x[im_num,:,:,:]
      im_pad = np.pad(im,((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
      im_col = im2col(im_pad,HH,WW,stride)
      filter_col = np.reshape(w,(F,-1))
      mul = im_col.dot(filter_col.T)
      # mul = im_col.dot(filter_col.T) + b
      out[im_num,:,:,:] = col2im(mul,H_prime,W_prime,1)
  cache = (x, w, b, conv_param)
  return out, cache

def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  x, w, b, conv_param = cache
  pad_num = conv_param['pad']
  stride = conv_param['stride']
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_prime = (H+2*pad_num-HH) // stride + 1
  W_prime = (W+2*pad_num-WW) // stride + 1

  dw = np.zeros(w.shape)
  dx = np.zeros(x.shape)
  db = np.zeros(b.shape)

  # We could calculate the bias by just summing over the right dimensions
  # Bias gradient (Sum on dout dimensions (batch, rows, cols)
  #db = np.sum(dout, axis=(0, 2, 3))

  for i in range(N):
      im = x[i,:,:,:]
      im_pad = np.pad(im,((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
      im_col = im2col(im_pad,HH,WW,stride)  #[OHxOW, ICxKHxKW]
      filter_col = np.reshape(w,(F,-1)).T

      dout_i = dout[i,:,:,:]
      dbias_sum = np.reshape(dout_i,(F,-1))  #[OC, OHxOW]
      dbias_sum = dbias_sum.T #[OHxoW, OC]

      #bias_sum = mul + b
      db += np.sum(dbias_sum,axis=0)
      dmul = dbias_sum #[OHxOW, OC]

      #mul = im_col * filter_col
      dfilter_col = (im_col.T).dot(dmul) #[ICxKHxKW, OC]
      dim_col = dmul.dot(filter_col.T) #[OHxOW, ICxKHxKW]

      dx_padded = col2im_back(dim_col,H_prime,W_prime,stride,HH,WW,C)
      dx[i,:,:,:] = dx_padded[:,pad_num:H+pad_num,pad_num:W+pad_num]
      dw += np.reshape(dfilter_col.T,(F,C,HH,WW))
  return dx, dw, db

def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(mul,h_prime,w_prime,C):
    """
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out

def col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c):
    """
    Args:
      dim_col: gradients for im_col,(h_prime*w_prime,hh*ww*c)
      h_prime,w_prime: height and width for the feature map
      strid: stride
      hh,ww,c: size of the filters
    Returns:
      dx: Gradients for x, (C,H,W)
    """
    H = (h_prime - 1) * stride + hh
    W = (w_prime - 1) * stride + ww
    dx = np.zeros([c,H,W])
    for i in range(h_prime*w_prime):
        row = dim_col[i,:]
        h_start = int((i / w_prime) * stride)
        w_start = int((i % w_prime) * stride)
        dx[:,h_start:h_start+hh,w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
    return dx