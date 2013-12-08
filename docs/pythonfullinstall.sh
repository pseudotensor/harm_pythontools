#test line - MAVARA
USEOTHER=0
USEKRAKEN=0
USEPFE=1

alias cp='cp'
alias mv='mv'

# see:
#http://docs.python.org/install/index.html

# add to ~/.bashrc :
# export PYTHONPATH=$BASE/lib/python/:$BASE/py/

if [ $USEOTHER -eq 1 ]
then
    SRCDIR=/data/othersrc/pythonstuff
    BASE=$HOME
fi
if [ $USEKRAKEN -eq 1 ]
then
    # For Kraken:
    SRCDIR=/lustre/scratch/$USER/tarballs
    BASE=/lustre/scratch/$USER/

    #source ~/setuppython27
    # trying to get MKL to work
    source ~/setuppython27icc
    module unload python/2.7.1
    module unload python/2.7.1-cnl
fi

if [ $USEPFE -eq 1 ]
then
    # For PFE:
    SRCDIR=/nobackup/$USER/tarballs/
    #BASE=/u/$USER/
    BASE=/nobackup/$USER/

fi



# make tools path as consistent with ~/.bashrc line:
mkdir -p $BASE/lib/python

# get tarballs, e.g.:
# cd $SCRDIR
# scp -rp jmckinne@ki-jmck.slac.stanford.edu:/data1/jmckinne/pythonstuff/tarballs .

# PYTHON
cd $SRCDIR
#tar xvzf Python-2.7.2.tgz
cd Python-2.7.2/
make clean
# on normal system using icc
#./configure --prefix=$BASE/
if [ $USEKRAKEN -eq 1 ]
then
    # worse:
    #export CC=cc
    # on normal system using icc
    ./configure --prefix=$BASE/
fi
if [ $USEPFE -eq 1 ]
then
#mkdir ~/bin/
#cd ~/bin/
#ln -s /nasa/intel/Compiler/2012.0.032/bin/icc cc
#export PATH=$HOME/bin/$PATH  # and add this to ~/.profile
## on PFE:
#cd $SRCDIR
#cd Python-2.7.2/
#LDFLAGS=/nasa/intel/Compiler/2012.0.032/mkl/lib/intel64/ -lmkl -lmkl_scalapack_lp64
#CPPFLAGS=-I/nasa/intel/Compiler/2012.0.032/mkl/include/
#./configure --without-gcc --prefix=$BASE/
# problem with MKL on PFE, so just stick with gcc and lapack/blas from netlib.
LDFLAGS=
CPPFLAGS=
rm -rf ~/bin/cc
./configure --prefix=$BASE/
fi

make
# ok that curses doesn't work, but no other packages should fail.
make install
# then logout-login to ensure python path set right because still loads old python even if which python points to new!
#       or at least check that new python really loads by loading it and checking what reversion it reports.

# add below exports to bash start-up script (maybe .bashrc or .profile)
export PYTHONPATH=$BASE/lib/python/:$BASE/py/
export PATH=$HOME/bin/:$BASE/bin:$PATH
export PYTHON_LIB=$BASE/lib/
export PYTHON_INC=$BASE/include/python2.7/
export LD_LIBRARY_PATH=$HOME/lib/:$BASE/lib/:$LD_LIBRARY_PATH
export LIBRARY_PATH=$BASE/lib/:$LIBRARY_PATH

if [ $USEPFE -eq 1 ]
then
cd $BASE/lib
ln -s python2.7 python
#mv $BASE/lib/python2.7/site.py $BASE/lib/python2.7/site.py.backup
mv /nobackupp8/jmckinn2/lib/python2.7/site.py /nobackupp8/jmckinn2/lib/asdf.py
fi

cd $SRCDIR
#tar xvzf setuptools-0.6c11.tar.gz
cd setuptools-0.6c11/
rm -rf build
python setup.py install --home=$BASE

# don't do below.
# mv $BASE/lib/python2.7/site.py.backup $BASE/lib/python2.7/site.py

# doesn't work on PFE for some reason
if [ $USEPFE -eq 0 ]
then
cd $SRCDIR
#tar xvzf python-dateutil-1.5.tar.gz
cd python-dateutil-1.5/
rm -rf build
python setup.py install --home=$BASE
fi

cd $SRCDIR
#tar xvzf yasm-1.1.0.tar.gz
cd yasm-1.1.0/
make clean
./configure --prefix=$BASE/
make
make install


cd $SRCDIR
#tar xvzf ipython-0.10.2.tar.gz
cd ipython-0.10.2/
rm -rf build
python setup.py install --home=$BASE

# don't do above if have mkl, skip...
#http://netlib.org/blas/
# if lapack exists, must be working (wasn't on ki-rh42 -- broken link even)
#http://www.netlib.org/lapack/
#svn co https://icl.cs.utk.edu/svn/lapack-dev/lapack/trunk
#cd trunk
#cp make.inc.example make.inc
# failed with gfortran: ../../librefblas.a: No such file or directory
# just removed /usr/lib64/liblapack* since wasn't on ki-jmck

cd $SRCDIR
# tar xvzf nose-0.11.4.tar.gz
cd nose-0.11.4
rm -rf build
python setup.py install --home=$BASE




if [ $USEPFE -eq 1 ]
then
cd $SRCDIR
wget http://www.netlib.org/blas/blas.tgz
cd BLAS
make
mkdir ~/lib/
cp blas_LINUX.a ~/lib/
#
cd $SRCDIR
wget http://www.netlib.org/lapack/lapack-3.5.0.tgz
tar xvzf lapack-3.5.0.tgz
cd lapack-3.5.0
cp $SRCDIR/BLAS/blas_LINUX.a librefblas.a
cp librefblas.a ~/lib/
cp make.inc.example make.inc
make
cp liblapack.a libtmglib.a ~/lib/
fi




if [ $USEPFE -eq 0 ]
then
############
# NON-PFE systems MKL

# Intel Studio if not already installed
cd /data/jon/
# tar xvzf c_studio_xe_2011_update2.tgz
cd c_studio_xe_2011_update2/
sh install.sh
cd /opt/intel
chmod -R a+rx .
# add to /etc/bashrc:
# source /opt/intel/bin/compilervars.sh intel64
# relogin as user

# MKL if not already installed
#http://software.intel.com/en-us/articles/non-commercial-software-download/
cd /data/jon/
# tar xvzf l_mkl_10.3.7.256.tgz
cd l_mkl_10.3.7.256
# NFJ6-FFWZM76K
sh install.sh
cd /opt/intel/
chmod -R a+rx composer_xe_2011_sp1/ 

#/opt/intel/composer_xe_2011_sp1.7.256

# latest MKL And numpy:
http://www.scipy.org/Installing_SciPy/Linux
# Intel link advisor:
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/


# DONE WITH MKL
###############################
fi

cd $SRCDIR
#tar xvzf numpy-1.6.1.tar.gz
cd numpy-1.6.1/
rm -rf build
# cp site.cfg.example site.cfg

# Take [mkl] section, copy to site.cfg, and then uncomment that section.
# For INTEL icc:
# with MKL just do:
# http://www.shocksolution.com/2008/09/installing-numpy-with-the-intel-math-kernel-library-mkl/
# emacs site.cfg and uncomment and edit [mkl] section
# e.g.
# OLD:
# [mkl]
# library_dirs = /opt/intel/mkl/10.0.3.020/lib/em64t
# include_dirs = /opt/intel/mkl/10.0.3.020/include
# lapack_libs = mkl_lapack
# mkl_libs = mkl, guide
# NEW:
#[mkl]
#library_dirs = /opt/intel/composer_xe_2011_sp1.7.256/mkl/lib/intel64
#include_dirs = /opt/intel/composer_xe_2011_sp1.7.256/mkl/include
#lapack_libs = mkl_scalapack_lp64
#mkl_libs = 


# for Kraken, uses libsci instead of mkl
# [mkl]
# library_dirs = /opt/intel/composer_xe_2011_sp1.8.273/mkl/lib/intel64/
# include_dirs = /opt/intel/composer_xe_2011_sp1.8.273/mkl/include/
# lapack_libs = mkl_scalapack_lp64
# mkl_libs =


# for PFE with MKL (not working)

# [mkl]
# library_dirs = /nasa/intel/Compiler/2012.0.032/mkl/lib/intel64/
# include_dirs = /nasa/intel/Compiler/2012.0.032/mkl/include/
# lapack_libs = mkl_scalapack_lp64
# mkl_libs =


# PFE without MKL
#[DEFAULT]
# REPLACE jmckinn2 by your $USER
#library_dirs = /usr/local/lib,/u/jmckinn2/lib
#include_dirs = /usr/local/include,/u/jmckinn2/include
#libraries = f77blas, cblas, atlas
#[lapack_opt]
#libraries = lapack, f77blas, cblas, atlas
#              

#

#
# emacs numpy/distutils/intelccompiler.py
# for intelem section:
#     cc_exe = 'icc -O2 -g -openmp -fPIC  -L/opt/intel/composer_xe_2011_sp1.8.273/mkl/lib/intel64/ -lmkl -axv'
#     cc_args = "-fPIC -DMKL_ILP64 "
#
# for Kraken, replace -L with what used above in [mkl] section.  Same for PFE.
# for PFE, remove -axv
#
if [ $USEKRAKEN -eq 1 ]
then
python setup.py config --compiler=intelem build_clib --compiler=intelem build_ext --compiler=intelem install --home=$BASE
fi

##########
## To use GCC:
if [ $USEPFE -eq 1 ]
then
#python setup.py install --home=$BASE
python setup.py config --compiler=unix --fcompiler=gnu95 install --home=$BASE
fi

# test numpy to see if anything really works (even compile):
cd $SRCDIR
# NOT WORKING for Kraken or PFE
python -c 'import numpy; numpy.test()'



####################
# OLD scipy
#Search on gridspec iself: http://matplotlib.sourceforge.net/api/gridspec_api.html
#led me to fact that one must install matplotlib 1.0 to get gridspec.
#I followed:  http://matplotlib.sourceforge.net/faq/installing_faq.html#install-from-git
#I have to ensure to do the below in the /usr/local/lib/python2.7/ directory:
#find . -type d -exec chmod -R a+r {} \; ; find . -type d -exec chmod -R a+x {} \; ; find . -type d -exec chmod -R g-s {} \;
#Then I can do stuff like: http://matplotlib.sourceforge.net/users/pyplot_tutorial.html
cd $SRCDIR
#tar xvzf python-scipy_0.9.0+dfsg1.orig.tar.gz
cd scipy-0.9.0.orig/
rm -rf build
if [ $USEPFE -eq 0 ]
then
python setup.py install --home=$BASE
fi

if [ $USEPFE -eq 1 ]
then
export LAPACK=$HOME/lib/liblapack
export LAPACK_SRC=$SRCDIR/lapack-3.5.0/
export BLAS=$HOME/lib/blas_LINUX.a
export BLAS_SRC=$SRCDIR/BLAS/
cp $SRCDIR/lapack-3.5.0/liblapack.a $SRCDIR/lapack-3.5.0/libtmglib.a .
cp $SRCDIR/BLAS/blas_LINUX.a librefblas.a
cp $SRCDIR/BLAS/blas_LINUX.a libblas.a
python setup.py config --compiler=unix --fcompiler=gnu95 install --home=$BASE
fi

#['setup.py', 'config', '--compiler=gcc', 'config_fc', '--fcompiler=gfortran', 'build_clib', '--compiler=gcc', '--fcompiler=gfortran', 'build_ext', '--compiler=gcc', '--fcompiler=gfortran', 'install', '--prefix=/nobackup/jmckinn2/']


# DON'T USE NEW SCIPY
if [ 1 -eq 0 ]
then
####################
# NEW scipy
# http://www.scipy.org/Download
cd $SRCDIR
#tar xvzf scipy-0.10.0.tar.gz
cd scipy-0.10.0
rm -rf build
python setup.py config --compiler=intelem --fcompiler=intelem build_clib --compiler=intelem --fcompiler=intelem build_ext --compiler=intelem --fcompiler=intelem install --home=$BASE
##########
fi



# matplotlib
cd $SRCDIR
# tar xvzf matplotlib.tgz
cd matplotlib/matplotlib
rm -rf build
if [ $USEPFE -eq 1 ]
then
python setup.py config --compiler=unix --fcompiler=gnu95 install --home=$BASE
fi
if [ $USEPFE -eq 0 ]
then
python setup.py install --home=$BASE
fi

if [ $USEPFE -eq 0 ] # pfe already has it as a module one can load
then
# ffmpeg
cd $SRCDIR
#tar xvzf ffmpeg-0.8.tar.gz
cd ffmpeg-0.8/
make clean
./configure --prefix=$BASE/
make
make install
make libinstall
make libainstall
fi

# try optimize.leastsq():
# http://www.scipy.org/scipy_Example_List#head-4c436ae0085d9a56056425d11abff4ccdd3d3620

