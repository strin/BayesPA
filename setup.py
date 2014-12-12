from distutils.core import setup, Extension

dir_medlda = 'medlda/OnlineGibbsMedLDA/'
pamedlda = Extension('pamedlda',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '1')],
                    include_dirs = [dir_medlda + 'inc', dir_medlda + 'inc/utils'],
                    libraries = ['boost_python'],
                    extra_compile_args=['-std=c++11', '-w'], 
                    sources = [dir_medlda + name for name in [
                                        "src/utils/ap.cpp",
                                        "src/utils/cholesky.cpp",
                                        "src/utils/cokus.cpp",
                                        "src/utils/InverseGaussian.cpp",
                                        "src/utils/Mapper.cpp",
                                        "src/utils/MVGaussian.cpp",
                                        "src/utils/objcokus.cpp",
                                        "src/utils/spdinverse.cpp",
                                        "src/utils/Document.cpp",
                                        "src/OnlineGibbsMedLDA.cpp",
                                        "src/OnlineGibbsMedLDAWrapper.cpp"
                                  ]])

setup(name='medlda',
      version='0.1',
      description='Online Maximum Entropy Discriminant Latent Dirichlet Allocation (Online MedLDA)',
      url='http://github.com/strin/BayesPA',
      author='Tianlin Shi',
      author_email='tianlinshi@gmail.com',
      license='GPL',
      packages=['medlda'],
      ext_modules = [pamedlda],
      zip_safe=False)
