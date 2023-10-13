#!/usr/bin/env bash

reg='(.*)([~<>=]{1}=)(.*)'

while IFS= read -r line
do

if [[ "$line" != *"#"* ]]
then

if [[ "$line" =~ $reg ]]
then

echo "${#BASH_REMATCH[@]}"
poetry add "${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}"

elif [[ "$line" == *"git+https"* ]]
then

poetry add "$line"

fi

fi

done < requirements.txt

