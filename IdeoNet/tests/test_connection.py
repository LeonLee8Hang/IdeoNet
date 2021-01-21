import pytest
import torch


# NOTE: Base class is not tested, propagate spike is tested in the linear layer due to initialization problems of the weights


##########################################################
# Test Connection: delayed propagation of spikes and spike conversion
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # shape (in feat, out feat, batch), dt, delay
        ((1, 4, 2), 1.0, 3)
    ],
)
def delayed_connection(request):
    from torch.nn.parameter import Parameter
    from ideonet.network.topology import Connection

    params = request.param
    connection = Connection(*params)
    connection.weight = Parameter(torch.Tensor(params[0][1], params[0][0]))
    connection.init_connection()
    return connection


# Test spike propagation
@pytest.mark.parametrize(
    "spikes_in, spikes_out",
    [
        (
            torch.zeros(1, 1, 2, dtype=torch.bool),
            torch.zeros(1, 4, 2, dtype=torch.float),
        ),
        (torch.ones(1, 1, 2, dtype=torch.bool), torch.ones(1, 4, 2, dtype=torch.float)),
        (
            torch.tensor([[[1, 0]]], dtype=torch.bool),
            torch.tensor([[[1, 0], [1, 0], [1, 0], [1, 0]]], dtype=torch.float),
        ),
    ],
)
def test_propagate_delayed(spikes_in, spikes_out, delayed_connection):
    r"""Checks if delayed spikes are properly converted and propagated."""
    spikes_in = delayed_connection.convert_spikes(spikes_in)
    conn_out = delayed_connection.propagate_spike(spikes_in)
    empty_out = torch.zeros_like(spikes_out)
    assert (conn_out == empty_out).all()

    # Loop over delay until output from connection
    empty_in = torch.zeros_like(spikes_in)
    steps = delayed_connection.delay_init.view(-1)[0].int() - 1
    for _ in range(steps):
        conn_out = delayed_connection.propagate_spike(empty_in)
    assert (conn_out == spikes_out).all()

    # Make sure no secondary spikes occur
    conn_out = delayed_connection.propagate_spike(empty_in)
    assert (conn_out == empty_out).all()


##########################################################
# Test Connection: instant propagation of spikes
##########################################################
@pytest.fixture(
    scope="function",
    params=[
        # shape (in feat, out feat, batch), dt, delay
        ((1, 4, 2), 1.0, 0)
    ],
)
def instant_connection(request):
    from torch.nn.parameter import Parameter
    from ideonet.network.topology import Connection

    params = request.param
    connection = Connection(*params)
    connection.weight = Parameter(torch.Tensor(params[0][1], params[0][0]))
    connection.init_connection()
    return connection


# Test spike propagation
@pytest.mark.parametrize(
    "spikes_in, spikes_out",
    [
        (
            torch.zeros(1, 1, 2, dtype=torch.bool),
            torch.zeros(1, 4, 2, dtype=torch.float),
        ),
        (torch.ones(1, 1, 2, dtype=torch.bool), torch.ones(1, 4, 2, dtype=torch.float)),
        (
            torch.tensor([[[1, 0]]], dtype=torch.bool),
            torch.tensor([[[1, 0], [1, 0], [1, 0], [1, 0]]], dtype=torch.float),
        ),
    ],
)
def test_propagate_instant(spikes_in, spikes_out, instant_connection):
    r"""Checks if instant spikes are properly converted and propagated."""
    spikes_in = instant_connection.convert_spikes(spikes_in)
    conn_out = instant_connection.propagate_spike(spikes_in)
    assert (conn_out == spikes_out).all()


##########################################################
# Test Linear Connection
##########################################################
lnr_in_feat = 5
lnr_out_feat = 10
lnr_batch = 2
lnr_in_shape = (lnr_batch, 1, lnr_in_feat)
lnr_out_shape = (lnr_batch, 1, lnr_out_feat, lnr_in_feat)
lnr_wgt_shape = (lnr_out_feat, lnr_in_feat)


@pytest.fixture(
    scope="function",
    params=[
        # in_features, out_features, batch_size, dt, delay, weight
        (lnr_in_feat, lnr_out_feat, lnr_batch, 1.0, 0, 0.5)
    ],
)
def linear_connection(request):
    from ideonet.network.topology import Linear

    params = request.param
    connection = Linear(*params[:-1])
    connection.reset_weights("constant", params[-1])
    return connection


@pytest.mark.parametrize(
    "spikes_in, weights, potential_out",
    [
        (
            torch.ones(*lnr_out_shape, dtype=torch.float),
            0.5,
            torch.ones(*lnr_out_shape, dtype=torch.float) * 0.5,
        ),
        (
            torch.ones(*lnr_out_shape, dtype=torch.float),
            1.0,
            torch.ones(*lnr_out_shape, dtype=torch.float) * 1.0,
        ),
        (
            torch.ones(*lnr_out_shape, dtype=torch.float),
            2.0,
            torch.ones(*lnr_out_shape, dtype=torch.float) * 2.0,
        ),
        (
            torch.ones(*lnr_out_shape, dtype=torch.float),
            3.0,
            torch.ones(*lnr_out_shape, dtype=torch.float) * 3.0,
        ),
    ],
)
def test_linear_activation_potential(
    spikes_in, weights, potential_out, linear_connection
):
    r"""Checks if the proper activation potential is calculated."""
    linear_connection.reset_weights("constant", weights)
    potential = linear_connection.activation_potential(spikes_in)
    assert (potential == potential_out).all()


@pytest.mark.parametrize(
    "spikes_in, trace_in, activation_out, trace_out",
    [
        (
            torch.ones(*lnr_in_shape, dtype=torch.bool),
            torch.ones(*lnr_in_shape, dtype=torch.float),
            torch.ones(*lnr_out_shape, dtype=torch.float) * 0.5,
            torch.ones(*lnr_out_shape, dtype=torch.float),
        )
    ],
)
def test_linear_forward(
    spikes_in, trace_in, activation_out, trace_out, linear_connection
):
    r"""Checks correct activation and trace increase."""
    activation, trace = linear_connection.forward(spikes_in, trace_in)
    assert (activation == activation_out).all()
    assert (trace == trace_out).all()


##########################################################
# Test Conv2d layer functions
##########################################################
conv_chan_in = 1
conv_chan_out = 2
conv_kernel = (3, 3)
conv_in_im = (10, 10)
conv_batch = 2
conv_out_im = (8, 8)

conv_in_shape = (conv_batch, conv_chan_in, *conv_in_im)
conv_out_shape = (
    conv_batch,
    conv_chan_in,
    *conv_out_im,
    conv_kernel[0] * conv_kernel[1],
)
conv_wgt_shape = (conv_chan_out, conv_chan_in, *conv_kernel)


@pytest.fixture(
    scope="function",
    params=[
        # in_channels, out_channels, kernel_size, im_dims, batch_size, dt, delay, weight
        (conv_chan_in, conv_chan_out, conv_kernel, conv_in_im, conv_batch, 1.0, 0, 0.5)
    ],
)
def conv2d_connection(request):
    from ideonet.network.topology import Conv2d

    params = request.param
    connection = Conv2d(*params[:-1])
    connection.reset_weights("constant", params[-1])
    return connection


@pytest.mark.parametrize(
    "spikes_in, weights, potential_out",
    [
        (
            torch.ones(
                conv_batch,
                conv_chan_in,
                conv_kernel[0] * conv_kernel[1],
                conv_out_im[0] * conv_out_im[1],
                dtype=torch.float,
            ),
            torch.ones(*conv_wgt_shape, dtype=torch.float) * 0.5,
            torch.ones(*conv_out_shape, dtype=torch.float) * 0.5,
        )
    ],
)
def test_conv2d_activation_potential(
    spikes_in, weights, potential_out, conv2d_connection
):
    conv2d_connection.weight.data = weights
    potential = conv2d_connection.activation_potential(spikes_in)
    assert (potential == potential_out).all()


@pytest.mark.parametrize(
    "spikes_in, trace_in, activation_out, trace_out",
    [
        (
            torch.ones(*conv_in_shape, dtype=torch.bool),
            torch.ones(*conv_in_shape, dtype=torch.float),
            torch.ones(*conv_out_shape, dtype=torch.float) * 0.5,
            torch.ones(*conv_out_shape, dtype=torch.float),
        )
    ],
)
def test_conv2d_forward(
    spikes_in, trace_in, activation_out, trace_out, conv2d_connection
):
    activation, trace = conv2d_connection.forward(spikes_in, trace_in)
    assert (activation == activation_out).all()
    assert (trace == trace_out).all()
