#!/bin/bash
# Author: Tomas Goldsack


fileid="1kOG8kxqxuTSBCaEZg20CdSz37Y2CJy3A"
filename="st2_dev-data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

fileid="1ZbOjv0t66NAt6SOCyFi0KHcafY-j1ud9"
filename="bert-base-8192.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

