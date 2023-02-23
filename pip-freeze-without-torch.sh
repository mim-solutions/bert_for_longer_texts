#!/bin/bash
pip list --format=freeze | grep -vE '^(torch|torchvision|torchaudio)'