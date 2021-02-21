#!/usr/bin/env python
# coding: utf-8

import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import sys
import json
import pretrained_networks
import websockets
import logging
import asyncio
import base64
import io


class Model:
    def __init__(self, network_pkl, truncation_psi):
        sc = dnnlib.SubmitConfig()
        sc.num_gpus = 1
        sc.submit_target = dnnlib.SubmitTarget.LOCAL
        sc.local.do_not_copy_source_files = True

        print('Loading networks from "%s"...' % network_pkl)
        self.__G, self.__D, self.__Gs = pretrained_networks.load_networks(network_pkl)
        self.__noise_vars = [var for name, var in self.__Gs.components.synthesis.vars.items() if name.startswith("noise")]

        self.__Gs_kwargs = dnnlib.EasyDict()
        self.__Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.__Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            self.__Gs_kwargs.truncation_psi = truncation_psi

    def weights_to_image(self, weights):
        rnd = np.random.RandomState(0)
        z = rnd.randn(1, *self.__Gs.input_shape[1:])  # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.__noise_vars})  # [height, width]
        images = self.__Gs.run(np.array([weights]), None, **self.__Gs_kwargs)  # [minibatch, height, width, channel]
        return images[0]
        # PIL.Image.fromarray(images[0], "RGB")


async def websocket_request(websocket, path):
    async for message in websocket:
        not_supported_error = {"error": "not supported"}
        try:
            data = json.loads(message)
            if "request" in data:
                request = data["request"]
                logger.info("request: %s" % request)
                if request == "weights_to_image":
                    weights = np.array(data["weights"], dtype=np.float32)
                    logger.info("weights shape: %s" % str(weights.shape))
                    image = model.weights_to_image(weights)
                    buffer = io.BytesIO()
                    PIL.Image.fromarray(image).save(buffer, format="png")
                    image = base64.b64encode(buffer.getvalue()).decode("ascii")
                    response = {"response": "image", "image": image, "format": "png", "encoding": "base64"}
                    await websocket.send(json.dumps(response))
                else:
                    await websocket.send(json.dumps(not_supported_error))
            else:
                await websocket.send(json.dumps(not_supported_error))
        except json.decoder.JSONDecodeError:
            await websocket.send(json.dumps(not_supported_error))
        except Exception as e:
            logging.error("exception: %s" % e)
            await websocket.send(json.dumps({"error": "internal error"}))


def configure_logger():
    params = {}
    params["format"] = "[%(asctime)s.%(msecs)03d][%(levelname)7s][%(name)6s] %(message)s"
    params["datefmt"] = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(**params)


configure_logger()
logger = logging.getLogger("neural")
logger.setLevel(logging.INFO)

logger.info("loading model")
model = Model("gdrive:networks/stylegan2-ffhq-config-f.pkl", 0.5)

logger.info("start WebSocket server")
start_server = websockets.serve(websocket_request, "0.0.0.0", 3395)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
