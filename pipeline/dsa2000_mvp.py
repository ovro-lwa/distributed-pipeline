import orca.transform.deconvolve
import orca.transform.imaging
import orca.transform.image_warp
import glob

# inputs
psffiles = glob.glob('/fast/claw/*intervals*psf.fits')
dirtyfiles = glob.glob('/fast/claw/*intervals*dirty.fits')
names = [n.split('-')[1] for n in psffiles]

#params
niter = 1000
wait = True

if __name__ == '__main__':
    results = []
    for (dirty, psf, name) in zip(dirtyfiles, psffiles, names):
        res = orca.transform.deconvolve.convert_and_deconvolve.delay(dirty, psf, name, niter=niter)
        res2 = orca.transform.image_warp.image_plane_correction.delay(res, plot=False)
        results.append(res2)

    print(f'{len(results)} tasks submitted')

    if wait:
        print('Waiting for tasks to finish...')
        try:
            unfinished = [res for res in results if res.status != 'SUCCESS']
            len0 = len(unfinished)
            while any(unfinished):
                if len0 > len(unfinished):
                    print(f'{len(unfinished)}...', end='')
                    len0 = len(unfinished)
                unfinished = [res for res in results if res.status != 'SUCCESS']
        except KeyboardInterrupt:
            print('exiting...')

    print('deconvolution complete. stacking images...')
    fits_list = [res.get() for res in results]
    res = orca.transform.imaging.stack_images.delay(fits_list, 'stacked.fits')

    print('stacking task submitted')
    print('waiting for stacking task to finish...')
    res.wait()

