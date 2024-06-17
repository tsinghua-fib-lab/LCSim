#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Project Directory: ${PROJECT_DIR}"
echo "Generate proto and grpc python package"

OUTPUT_PATH=${PROJECT_DIR}/entity/gen
INCLUDE_PATH=${PROJECT_DIR}/protos
FILE_PREFIX=${PROJECT_DIR}/protos
# 请修改待生成_pb2.py文件的proto文件列表
FILES="\
    ${FILE_PREFIX}/geo/geo.proto \
    ${FILE_PREFIX}/map/light.proto \
    ${FILE_PREFIX}/map/map.proto \
    ${FILE_PREFIX}/agent/agent.proto \
    ${FILE_PREFIX}/agent/trip.proto \
    ${FILE_PREFIX}/routing/routing.proto \
"

rm -r ${OUTPUT_PATH} 2>/dev/null || true
mkdir -p ${OUTPUT_PATH}

protoc --python_out=${OUTPUT_PATH} --pyi_out=${OUTPUT_PATH} -I ${INCLUDE_PATH} ${FILES}
protol --create-package --in-place --python-out ${OUTPUT_PATH} \
    protoc --proto-path=${INCLUDE_PATH} ${FILES}