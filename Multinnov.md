# Depthai diagnostic scripts

This version of the depthai python library has been repuposed to server as a disagnostic script for any depthai device

### Dependances installation
The scripts depend on any other dependancies. Make sure to install the requirements like this :
```sh
sudo apt install python3-pip
cd [/path/to/this/folder]
sudo python3 -m pip install -r requirements.txt
```
    
### Execution and Disgnostic
For any device provided by luxonis, tests can be runned via  the scripts present in the **utilities** folder.
* The cam_test.py is a well rounded test file that autodetect all cameras peresnt and on any luxonis board. It will also print out fps, connection speed and other usefull datas. To use it just invoque 
```sh
cd [/path/to/utilities]
./cam_test.py
```
The only downside is that you have to have a gui in order to run it and it does not detect IMUs.
* The cam_test_without_show.py on the other hand works without gui and saves the images captured in a directory. 
It captures the frames for a duration of 20s and produces a log file in the given directory. If no directory is given, all the data will be in **/tmp/som_report**. To use it, just invoque : 
```sh
cd [/path/to/utilities]
./cam_test_without_show.py --report-path /path/to/report/folder
```
Make sure you have the right to the folder give  though. If it does not exist, it will be created.

