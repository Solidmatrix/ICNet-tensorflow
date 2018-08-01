#!/bin/bash

python convert_to_uff.py tensorflow \
--input-file icnet_model_conv6_cls.pb \
-o ./icnet_model_conv6_cls.uff \
-O conv6_cls/BiasAdd
