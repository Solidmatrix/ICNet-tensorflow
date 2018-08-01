#!/bin/bash

python convert_to_uff.py tensorflow \
--input-file icnet_model.pb \
-o ./icnet_model.uff \
-O conv6_cls/BiasAdd