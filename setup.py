from distutils.core import setup
import asm_protocol
setup(
    name='ASM Protocol',
    version=asm_protocol.__version__,
    description='ASM Packet Protocol',
    author='UC San Diego Engineers for Exploration',
    author_email='e4e@eng.ucsd.edu',
    packages=['asm_protocol'],
    install_requires=['numpy']
)
