import SimpleITK as sitk
import numpy as np
from image_coord import image_coord
from rect import rect
import os

from PIL import Image
import base64
import cv2
import dict_helper
import random

def read_key_value_pairs(file):
    dict = {}
    with open(file) as myfile:
        for line in myfile:
            key, value = line.partition('=')[::2]
            dict[key.strip()] = value.strip()
    return dict

# getting dependcy error: ImportError: libtbb.so.12: cannot open shared object file: No such file or directory
#import itk

def save_image(img, path):
    #print(f'save_image(img, {path})')
    sitk.WriteImage(img, path, True)  # useCompression:True

def read_image(path):
    #print(f'read_image({path})')
    return sitk.ReadImage(path)

def read_slice(mha_file_path, slice_index):
    image_3d = read_image(mha_file_path)
    # Get the size of the image (width, height, depth)
    size = image_3d.GetSize()

    # Extract the desired slice
    slice_2d = image_3d[:, :, slice_index]

    return slice_2d

# had issue with itk library, but rolled back to sitk
# def read_slice(mha_file_path, slice_index, output_pixel_type):

#     # Define the pixel type and dimension of the 3D image
#     PixelType = itk.ctype(output_pixel_type)
#     Dimension = 3
#     ImageType = itk.Image[PixelType, Dimension]

#     # Read the image metadata to get the size and other properties
#     image = itk.ImageFileReader[ImageType].New(FileName=mha_file_path)
#     image.UpdateOutputInformation()

#     # Get the size of the image
#     image_size = image.GetOutput().GetLargestPossibleRegion().GetSize()

#     # Define the region to extract (a single slice)
#     start = [0, 0, slice_index]
#     size = [image_size[0], image_size[1], 0]
#     region = itk.ImageRegion[Dimension]()
#     region.SetIndex(start)
#     region.SetSize(size)

#     # Extract the slice
#     ExtractorType = itk.ExtractImageFilter[ImageType, itk.Image[PixelType, 2]]
#     extractor = ExtractorType.New()
#     extractor.SetInput(image.GetOutput())
#     extractor.SetExtractionRegion(region)
#     extractor.SetDirectionCollapseToSubmatrix()  # This is required to handle the dimensionality reduction
#     extractor.Update()

#     # Get the slice as a 2D image
#     slice_image = extractor.GetOutput()

#     return slice_image

def read_slice_float_as_np(mha_file_path, slice_index):

    # read itk image
    slice_image = read_slice(mha_file_path, slice_index)

    # Convert the slice to a numpy array
    slice_array = sitk.GetArrayFromImage(slice_image)

    return slice_array.astype(np.float32)


    





def print_image_prop(img, name):
    print('image:', name)
    print('================')
    print('origin = ', np.array(img.GetOrigin()))
    print('spacing =', np.array(img.GetSpacing()))
    print('size = ', np.array(img.GetSize()))
    print('direction = ', np.array(img.GetDirection()))
    
def sample_image(img_path, grid_coord_w, margin=3, defaultPixelValue=0, interpolator=sitk.sitkLinear, sampled_image_path=None):
    #print(f'sample_image({img_path}, {grid_coord_w}, {margin}, {defaultPixelValue}, {interpolator}, {sampled_image_path})')
    img = read_image_partial(img_path, grid_coord_w, margin)
    #img = read_image(img_path)
    #print_image_prop(img, 'img')

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(grid_coord_w.spacing.tolist())
    resample.SetSize(grid_coord_w.size.tolist())
    resample.SetOutputDirection(grid_coord_w.direction)
    #resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(grid_coord_w.origin.tolist())
    #resample.SetTransform(sitk.Transform()) # this is a unit transform... I think.
    resample.SetDefaultPixelValue(defaultPixelValue)
    resample.SetInterpolator(interpolator)

    img_sampled = resample.Execute(img)
    #print_image_prop(img_sampled, 'img_sampled')

    # save image
    if sampled_image_path:
        save_image(img_sampled, sampled_image_path)

    return img_sampled


def sample_image2(img, grid_coord_w, margin=3, defaultPixelValue=0, interpolator=sitk.sitkLinear, sampled_image_path=None):
    #print(f'sample_image(img, {grid_coord_w}, {margin}, {defaultPixelValue}, {interpolator}, {sampled_image_path})')
    #print_image_prop(img, 'img')

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(grid_coord_w.spacing.tolist())
    resample.SetSize(grid_coord_w.size.tolist())
    resample.SetOutputDirection(grid_coord_w.direction)
    #resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(grid_coord_w.origin.tolist())
    #resample.SetTransform(sitk.Transform()) # this is a unit transform... I think.
    resample.SetDefaultPixelValue(defaultPixelValue)
    resample.SetInterpolator(interpolator)

    img_sampled = resample.Execute(img)
    #print_image_prop(img_sampled, 'img_sampled')

    # save image
    if sampled_image_path:
        save_image(img_sampled, sampled_image_path)

    return img_sampled


def get_image_coord(img_path):
    
    # image file reader
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_path)

    # read the image information without reading the bulk data, compute ROI start and size and read it.
    file_reader.ReadImageInformation()

    return image_coord(size=file_reader.GetSize(), origin=file_reader.GetOrigin(), spacing=file_reader.GetSpacing(), direction=file_reader.GetDirection())

def get_image_coord_from_itkImage(itkImage):
    return image_coord(size=itkImage.GetSize(), origin=itkImage.GetOrigin(), spacing=itkImage.GetSpacing(), direction=itkImage.GetDirection())




def read_image_partial(img_path, grid_coord_w, margin):

    #print(f'read_image_partial({img_path}, {grid_coord_w}, {margin})')

     # image file reader
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_path)

    # read the image information without reading the bulk data, compute ROI start and size and read it.
    file_reader.ReadImageInformation()

    img_coord_full = image_coord(size=file_reader.GetSize(), origin=file_reader.GetOrigin(), spacing=file_reader.GetSpacing(), direction=file_reader.GetDirection())
    #print('img_coord_full=', img_coord_full)

    # image index of the grid origin
    grid_org_imgI = img_coord_full.w2I(grid_coord_w.origin).astype(int)
    #print('grid_org_imgI=', grid_org_imgI)

    # grid size w.r.t image I
    grid_size_imgI = np.round(grid_coord_w.size_phys()/img_coord_full.spacing).astype(int)
    #grid_size_imgI = np.round((grid_coord_w.size * grid_coord_w.spacing)/img_coord_full.spacing).astype(int)
    #print('grid_size_imgI=', grid_size_imgI)

    # grid rect in the image I coordinates
    grid_rect_I = rect(grid_org_imgI, grid_org_imgI+grid_size_imgI)
    #print('grid_rect_I=', grid_rect_I)

    # expand the grid rect by 3
    read_rect_I = grid_rect_I.expand(margin)
    #print('read_rect_I expanded=', read_rect_I)

    # make sure the read_rect is within the image rect
    read_rect_I = read_rect_I.intersect(img_coord_full.rect_I())
    #print('read_rect_I=', read_rect_I)
    
    start_index = read_rect_I.low
    extract_size = read_rect_I.size()
    #print('start_index=', start_index)
    #print('extract_size=', extract_size)

    file_reader.SetExtractIndex(start_index.tolist())
    file_reader.SetExtractSize(extract_size.tolist())

    img = file_reader.Execute()
    #print_image_prop(img, 'img (partial)')

    return img

def get_grid_list_to_cover_rect(rect_w, grid_size, grid_spacing, n_border_pixels):
    #valid_grid_size = 64 - n_border_pixels * 2
    valid_grid_size = grid_size - n_border_pixels * 2
    valid_grid_size_mm = valid_grid_size * grid_spacing
    #print('valid_grid_size=', valid_grid_size)
    #print('valid_grid_size_mm=', valid_grid_size_mm)

    #print('rect_w=', rect_w)
    #print('rect_w.size()=', rect_w.size())

    #print('num of blocks to inference (fraction) =', rect_w.size()/valid_grid_size_mm)

    # round up to ints
    #N_sub_images = np.array(np.floor(organ_rect_w.size()/valid_grid_size_mm)+[1.0]*3).astype(int)
    N_sub_images = np.ceil(rect_w.size()/valid_grid_size_mm).astype(int)

    #print('num of sub images =', N_sub_images)

    # first grid (i=0, j-0, k=0)
    grid_coord_000 = None

    list = []

    # block to block distance (block size)
    block_spacing_u = [1.0, 1.0, 1.0] / N_sub_images
    #print('block_spacing_u=', block_spacing_u)
    for k in range(N_sub_images[2]):
        for j in range(N_sub_images[1]):
            for i in range(N_sub_images[0]):
                #print('======================')
                #print(f'rect_sub[{i}][{j}][{k}]')
                
                block_origin_u = block_spacing_u * [i, j, k]
                block_center_u = block_origin_u + block_spacing_u / 2.0
                #print('block_center_u=', block_center_u)

                # convert u to w (this is valid only when the direction is unity)
                block_center_w = rect_w.low + block_center_u * rect_w.size()
                #print('block_center_w=', block_center_w)

                # sampling grid when the grid is placed at the block center
                grid_org_w = block_center_w - [grid_size * grid_spacing / 2.0]*3
                grid_coord = image_coord(origin=grid_org_w, size=[grid_size]*3, spacing=[grid_spacing]*3)
                #print('grid_org_w=', grid_org_w)
                #print('grid_coord=', grid_coord)

                # keep the first grid coordinate 
                if grid_coord_000 is None:
                    grid_coord_000 = grid_coord
                    #print('grid_coord_000=', grid_coord_000)

                # the nearest sampled image pixel index of the current grid origin (i.e, the grid origin w.r.t to the first grid)
                grid_org_wrt_grid000_I = grid_coord_000.w2I(grid_org_w)
                #print('grid_org_wrt_grid000_I=', grid_org_wrt_grid000_I)

                # adjust the origin such that the grid orgin is aligned to first grid orgin
                grid_org_w = grid_coord_000.I2w(grid_org_wrt_grid000_I)
                grid_coord = image_coord(origin=grid_org_w, size=[grid_size]*3, spacing=[grid_spacing]*3)
                #print('grid_org_w=', grid_org_w)
                #print('grid_coord=', grid_coord)

                # tuple of grid_coord, grid_org_wrt_grid000_I
                # grid_coord is the sampling grid
                # grid_org_wrt_grid000_I is the first pixel index in the whole segment image
                list.append((grid_coord, grid_org_wrt_grid000_I))
    return list

def transfer_bbox_from_w_to_oI(img_mhd, str_info_file):
    
    # ObjectType = Image
    # NDims = 3
    # BinaryDataByteOrderMSB = False
    # CompressedData = False
    # TransformMatrix = -1 0 0 0 -1 0 0 0 1
    # Offset = 325 325 -147.5
    # CenterOfRotation = 0 0 0
    # ElementSpacing = 1.269531 1.269531 2.5
    # DimSize = 512 512 136
    # AnatomicalOrientation = ???
    # ElementSize = 1.269531 1.269531 2.5
    # ElementType = MET_SHORT
    # ElementDataFile = img.raw
    #print('img_mhd=', img_mhd)
    img_header = read_key_value_pairs(img_mhd)
    direction = [float(s) for s in img_header['TransformMatrix'].strip().split(' ')]
    origin = [float(s) for s in img_header['Offset'].strip().split(' ')]
    spacing = [float(s) for s in img_header['ElementSpacing'].strip().split(' ')]
    size = [float(s) for s in img_header['DimSize'].strip().split(' ')]
    
    ct_coord_full = image_coord(size=size, origin=origin, spacing=spacing, direction=direction)
    #print('ct_coord_full=', ct_coord_full)

    # read the bounding box
    dict = read_key_value_pairs(str_info_file)
    bbox_w = [float(s) for s in dict['bbox'].split(',')]

    #print('bbox_w=', bbox_w)
    minx, maxx, miny, maxy, minz, maxz = bbox_w
        
    low_w=np.array([minx, miny, minz]).astype(float)
    high_w=np.array([maxx, maxy, maxz]).astype(float)
    #print('low_w=', low_w)
    #print('high_w=', high_w)

    #print('transfering the bbow_w to image_o coordiantes')        
    low_o=ct_coord_full.w2o(low_w)
    high_o= ct_coord_full.w2o(high_w)
    #print('low_o=', low_o)
    #print('high_o=', high_o)
    #print('low_o.shape=', low_o.shape)
    #print('high_o.shape=', high_o.shape)

    mat = np.array([low_o, high_o])
    #print('mat=', mat)
    #print('mat.shape=', mat.shape)

    low_o = np.min(mat, axis=0)
    high_o = np.max(mat, axis=0)
    #print('low_o=', low_o)
    #print('high_o=', high_o)

    low_I = ct_coord_full.o2I(low_o)
    high_I = ct_coord_full.o2I(high_o)
    #print('low_I=', low_I)
    #print('high_I=', high_I)
    
    # append to the strinfo file
    line1 = f'bbox_o={low_o[0]},{high_o[0]},{low_o[1]},{high_o[1]},{low_o[2]},{high_o[2]}\n'
    line2 = f'bbox_I={low_I[0]},{high_I[0]},{low_I[1]},{high_I[1]},{low_I[2]},{high_I[2]}\n'
    #print('appending bbox_o to str_info_file')
    #print('line1=', line1)
    #print('line2=', line2)
    #print('str_info_file=', str_info_file)
    with open(str_info_file, 'a') as file:
        file.write(line1)
        file.write(line2)

def reset_image_orgin_and_direction(img_file):
        img = read_image(img_file)
        # reset origin & diretion
        img.SetOrigin([0.0, 0.0, 0.0])
        img.SetDirection([1.0, 0.0, 0.0, 
                          0.0, 1.0, 0.0, 
                          0.0, 0.0, 1.0])
        save_image(img, img_file)

def reset_image_origin_direction_and_appended_bbox_oI_for_new_downloads(downloaded_files):
    img_mhd = downloaded_files['image_mhd']
    # newly downloaded structure files
    for file in downloaded_files["structure_files_downloaded"]:
        _, ext = os.path.splitext(file)
        if ext.lower() == '.mha':
            reset_image_orgin_and_direction(file) # reset the origin and direction if a new structure.mha file is downloaded.
        elif ext.lower() == '.info':
            transfer_bbox_from_w_to_oI(img_mhd, file) # append bbox_o and bbox_I to the info file for a newly downloaded structure info file.
        else:
            pass
    # newly downloaded image files
    for file in downloaded_files["image_files_downloaded"]:
        _, ext = os.path.splitext(file)
        if ext.lower() == '.mha':
            reset_image_orgin_and_direction(file) # reset the origin and direction if a new image.mha file is downloaded.
            
def mhd_image_files_exist(mhd):
        if not os.path.exists(mhd):
            print('file not found:'+mhd)
            return False
        mha = mhd.replace('.mhd', '.mha')
        if not os.path.exists(mha):
            print('file not found:'+mha)
            return False
        info = mhd.replace('.mhd', '.info')
        if not os.path.exists(info):
            print('file not found:'+info)
            return False
        return True

   
def save_nparray_as_itk_image_ubyte(np_image, coord, out_file):

    # cast & conver to sitkImage to save
    itk_image = sitk.GetImageFromArray(np_image.astype(np.ubyte))

    # copy image properties
    itk_image.SetSpacing(coord.spacing)
    itk_image.SetOrigin(coord.origin)
    itk_image.SetDirection(coord.direction)

    # save image
    sitk.WriteImage(itk_image, out_file, True)  # useCompression:True

    return itk_image

def find_COM_of_binary_image(image_path):
    # Load the binary image

    binary_image = sitk.ReadImage(image_path)

    # If your binary image is not labeled (i.e., it's truly binary with values 0 and 1),
    # you can use it directly with the assumption that 1's represent your object.
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()

    # Execute the filter on the binary image
    label_shape_filter.Execute(binary_image)

    # Get the label of the object, assuming your object of interest is labeled as 1
    object_label = 1

    # Get the centroid for the object, which serves as the center of mass
    # the center of mass is w.r.t to the world coordinate system
    center_of_mass_w = label_shape_filter.GetCentroid(object_label)

    print(f"Center of Mass for the object labeled {object_label}: {center_of_mass_w}")

    return center_of_mass_w

def extract_and_resample_slice(image, center_of_mass_w, orientation, output_size=[256, 256], output_spacing=[1.0, 1.0], defaultPixelValue=0, interpolator=sitk.sitkLinear, output_pixel_type=sitk.sitkFloat32):
    """
    Extract and resample a slice from the 3D image.
    
    :param image: The 3D SimpleITK image.
    :param center_of_mass: The center of mass in physical coordinates (w).
    :param orientation: 'axial', 'sagittal', or 'coronal'.
    :param output_size: The desired output image size.
    :param output_spacing: The desired output image spacing.
    :return: Resampled 2D SimpleITK image.
    """
    # append dummy
    output_size = [output_size[0], output_size[1], 1]
    output_spacing = [output_spacing[0], output_spacing[1], 1.0]

    # Define resampling parameters
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputPixelType(output_pixel_type)
    resampler.SetDefaultPixelValue(defaultPixelValue)

    # w_H_imgo
    img_coord = get_image_coord_from_itkImage(image)
    w_H_imgo = img_coord.w_H_imgo()
    print(w_H_imgo)

    # com_w
    com_w = np.ones(4)
    com_w[:3] = np.array(center_of_mass_w)
    print(com_w)

    # com_imgo
    imgo_H_w = np.linalg.inv(w_H_imgo)
    com_imgo = imgo_H_w @ com_w
    print(com_imgo)       

    # grid size
    grid_size = np.array(output_size)
    grid_spacing = np.array(output_spacing)
    grid_size_phy = grid_size * grid_spacing
    print(grid_size_phy)
    
    grid_half_width = grid_size_phy[0]/2.0
    grid_half_height = grid_size_phy[1]/2.0
    # Extract slice according to orientation
    if orientation == 'axial':
       
        # grid origin in imgo
        vec_center_to_grido = np.array([-grid_half_width, -grid_half_height, 0.0])
        grido_imgo = com_imgo[:3] + vec_center_to_grido
        print(grido_imgo)

        #imgo_H_grido
        imgo_H_grido = np.identity(4)
        imgo_H_grido[:3, 3] = grido_imgo
        print(imgo_H_grido)

    elif orientation == 'sagittal':
        # grid origin in imgo
        vec_center_to_grido = np.array([0.0, -grid_half_width, grid_half_height])
        grido_imgo = com_imgo[:3] + vec_center_to_grido
        print(grido_imgo)

        #imgo_H_grido
        imgo_H_grido = np.array(
            [[0.0, 0.0, -1.0, grido_imgo[0]],
            [1.0, 0.0, 0.0, grido_imgo[1]],
            [0.0, -1.0, 0.0, grido_imgo[2]],
            [0.0, 0.0, 0.0, 1.0]])
        print(imgo_H_grido)
        
    elif orientation == 'coronal':
       # grid origin in imgo
        vec_center_to_grido = np.array([-grid_half_width, 0.0, grid_half_height])
        grido_imgo = com_imgo[:3] + vec_center_to_grido
        print(grido_imgo)

        #imgo_H_grido
        imgo_H_grido = np.array(
            [[1.0, 0.0, 0.0, grido_imgo[0]],
            [0.0, 0.0, 1.0, grido_imgo[1]],
            [0.0, -1.0,0.0, grido_imgo[2]],
            [0.0, 0.0, 0.0, 1.0]])
        print(imgo_H_grido)

    # w_H_grido
    w_H_grido = w_H_imgo @ imgo_H_grido
    print(w_H_grido)
    
    # extract grido and direction in w
    grido_w = w_H_grido[:3, 3]
    grid_direction_w = w_H_grido[:3, :3].reshape(-1)
    print(grido_w)
    print(grid_direction_w)

    # same orientation as the base image
    resampler.SetOutputDirection(grid_direction_w)
    resampler.SetOutputOrigin(grido_w)
    
    # Execute resampling
    resampled_slice = resampler.Execute(image)

    # return as 2D image
    return resampled_slice[:,:,0]

def extract_slices_for_overlay(img, str_img, seg_img, slice_center_w, orientation, output_size, output_spacing, draw_as_contours, out_dir):
    
    ret = {}

    # extract base image
    img_2d = extract_and_resample_slice(img, slice_center_w, orientation, output_size, output_spacing, defaultPixelValue=-1000)
    img_2d = apply_AHE_and_cast_to_uchar(img_2d)
    ret['img_png'] = os.path.join(out_dir,  f'{orientation}.img.png')
    save_image(img_2d, ret['img_png'])
    ret['img_mha'] = os.path.join(out_dir,  f'{orientation}.img.mha')
    save_image(img_2d, ret['img_mha'])

    # extract structure image
    str_2d = extract_and_resample_slice(str_img, slice_center_w, orientation, output_size, output_spacing, defaultPixelValue=0)
    str_2d = apply_rescale_and_cast_to_uchar(str_2d)
    ret['str_png'] = os.path.join(out_dir,  f'{orientation}.str.png')
    save_image(str_2d, ret['str_png'])
    ret['str_mha'] = os.path.join(out_dir,  f'{orientation}.str.mha')
    save_image(str_2d, ret['str_mha'])

    # extract segment image
    seg_2d = extract_and_resample_slice(seg_img, slice_center_w, orientation, output_size, output_spacing, defaultPixelValue=0)
    seg_2d = apply_rescale_and_cast_to_uchar(seg_2d)
    ret['seg_png'] = os.path.join(out_dir,  f'{orientation}.seg.png')
    save_image(seg_2d, ret['seg_png'])
    ret['seg_mha'] = os.path.join(out_dir,  f'{orientation}.seg.mha')
    save_image(seg_2d, ret['seg_mha'])

    # make overlay (img, structre)
    if draw_as_contours:
        str_overlay = overlay_mask_as_contours(img_2d, str_2d, 255, mask_out_color=[0, 255, 0])
    else: 
        str_overlay = overlay_mask(img_2d, str_2d, 255, mask_out_color=[0, 255, 0])
    ret['str_overlay'] = os.path.join(out_dir, f'{orientation}_overlay.str.png')
    save_image(str_overlay, ret['str_overlay'])

    # make overlay (img, segmentation)
    if draw_as_contours:
        seg_overlay = overlay_mask_as_contours(img_2d, seg_2d, 255, mask_out_color=[255, 0, 0])
    else: 
        seg_overlay = overlay_mask(img_2d, seg_2d, 255, mask_out_color=[255, 0, 0])
    ret['seg_overlay'] = os.path.join(out_dir, f'{orientation}_overlay.seg.png')
    save_image(seg_overlay, ret['seg_overlay'])

    return ret



    
def apply_AHE_and_cast_to_uchar(img):
    return cast_to_uchar(rescale_intensity(AHE(img)))

def apply_rescale_and_cast_to_uchar(img):
    return cast_to_uchar(rescale_intensity(img))

def AHE(img, alpha = 0.3, beta = 0.3):
    equalization_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
    equalization_filter.SetAlpha(alpha)  # Adjust these parameters as needed
    equalization_filter.SetBeta(beta)
    return equalization_filter.Execute(img)

def CED(img, th_low=0.0, th_high=1.0, gaussian_variance=[2.0, 2.0]):
    return sitk.CannyEdgeDetection(img, lowerThreshold=th_low, upperThreshold=th_high, variance=gaussian_variance)
    # lowerThreshold and upperThreshold are the thresholds for the hysteresis thresholding. You might need to adjust these values based on your specific image for optimal edge detection results.
    # variance is used to specify the size of the Gaussian filter for smoothing the image before edge detection. Larger values produce more smoothing, which can be useful for noisy images.

def rescale_intensity(img, out_min = 0,  out_max = 255):
    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(out_min)
    rescaler.SetOutputMaximum(out_max)
    return rescaler.Execute(img)

def cast_to_uchar(img):
    return sitk.Cast(img, sitk.sitkUInt8)

def cast_to_float32(img):
    return sitk.Cast(img, sitk.sitkFloat32)

def pack2rgb(img1, img2, img3):
    # Convert SimpleITK images to NumPy arrays
    img1_np = sitk.GetArrayFromImage(img1)
    img2_np = sitk.GetArrayFromImage(img2)
    img3_np = sitk.GetArrayFromImage(img3)

    # Stack arrays to create an RGB image
    rgb_np = np.stack([img1_np, img2_np, img3_np], axis=-1)

    # Optionally convert back to a SimpleITK image
    return sitk.GetImageFromArray(rgb_np, isVector=True)

def overlay_mask(img, mask, mask_forground = 255, mask_out_color=[255, 0, 0]):
    
    # threshould the mask image at the half of the mask_forground 
    mask = sitk.BinaryThreshold(mask, lowerThreshold=0, upperThreshold=mask_forground/2, insideValue=0, outsideValue=255)

    # np arrays
    img_np = sitk.GetArrayFromImage(img)
    mask_np = sitk.GetArrayFromImage(mask)
    
    # Stack arrays to create an RGB image
    rgb_np = np.stack([img_np, img_np, img_np], axis=-1)

    # override with red color
    rgb_np[mask_np == 255] = mask_out_color

    # convert back to a SimpleITK image
    return sitk.GetImageFromArray(rgb_np, isVector=True)

def overlay_mask_as_contours(img, mask, mask_forground=255, mask_out_color=[255, 0, 0]):
    # Convert SimpleITK images to NumPy arrays
    img_np = sitk.GetArrayFromImage(img)
    mask_np = sitk.GetArrayFromImage(mask)

    # Threshold the mask image at half of the mask_forground
    _, mask_np = cv2.threshold(mask_np, mask_forground / 2, mask_forground, cv2.THRESH_BINARY)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Stack arrays to create an RGB image
    rgb_np = np.stack([img_np, img_np, img_np], axis=-1)

    # Draw contours on the RGB image
    thickness = 2
    cv2.drawContours(rgb_np, contours, -1, mask_out_color, thickness)

    # convert back to a SimpleITK image
    return sitk.GetImageFromArray(rgb_np, isVector=True)

def make_grid_images(image_files, single_image_size , n_col, n_row, out_file):

    W = single_image_size[0]
    H = single_image_size[1]

    # grid image
    grid_image = Image.new('RGB', (W*n_col, H*n_row))

    # paste images
    for row in range(n_row):
        for col in range(n_col):
            i = col + row * n_col
            img_file = image_files[i]
            img = Image.open(img_file)
            grid_image.paste(img, (col*W, row*H))
    
    grid_image.save(out_file)

    return

    
def create_seg_overlay_images(img_file, str_file, seg_file, out_img_size=[256, 256], out_img_spacing=[0.5, 0.5], out_dir=None):

    if out_dir == None:
        out_dir = os.path.dirname(seg_file)

    # find the center of structure
    COM = find_COM_of_binary_image(str_file)

    img = read_image(img_file)
    str_img = read_image(str_file)
    seg_img = read_image(seg_file)

    output_size = out_img_size
    output_spacing = out_img_spacing

    # axial, sagittal, coronal
    draw_as_contours = True
    ax_files = extract_slices_for_overlay(img, str_img, seg_img, slice_center_w=COM, orientation='axial', output_size=output_size, output_spacing=output_spacing, draw_as_contours=draw_as_contours, out_dir=out_dir)
    sg_files = extract_slices_for_overlay(img, str_img, seg_img, slice_center_w=COM, orientation='sagittal', output_size=output_size, output_spacing=output_spacing,draw_as_contours=draw_as_contours,  out_dir=out_dir)
    cr_files = extract_slices_for_overlay(img, str_img, seg_img, slice_center_w=COM, orientation='coronal', output_size=output_size, output_spacing=output_spacing,draw_as_contours=draw_as_contours, out_dir=out_dir)

    # make one big overlay image
    overlay_files = [ax_files['seg_overlay'],sg_files['seg_overlay'],cr_files['seg_overlay'],
                     ax_files['str_overlay'],sg_files['str_overlay'],cr_files['str_overlay']]
    
    overlay_grid_file = os.path.join(out_dir, 'overlay.grid.png')
    make_grid_images(overlay_files, single_image_size=output_size, n_col = 3, n_row = 2, out_file = overlay_grid_file )

    # return file names
    files = {
            'overlay_grid_png': overlay_grid_file,
            'ax_files': ax_files,
            'sg_files': sg_files,
            'cr_files': cr_files
        }
    return files

def load_as_base64(file_name):
    # Open the file in binary mode and read its contents
    with open(file_name, 'rb') as file:
        binary_data = file.read()
    byte_string = base64.b64encode(binary_data)
    base64_string = byte_string.decode('utf-8')
    return base64_string

# Assuming 'binary_image' is your input 3D image and 'center_of_mass' is the center of mass in physical space

# You can now save or process these slices further as needed


def extract_largest_connected_compoment(binary_input_image, num_of_components=1):

    # select the largest object
    connected_component_image = sitk.ConnectedComponent(binary_input_image)
    print('type(connected_components)=', type(connected_component_image))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected_component_image)

    label_size_list = []
    for label in stats.GetLabels():
        print('================')
        print('label=', label)
        print('type(label)=', type(label))
        n_pixels = stats.GetNumberOfPixels(label)
        print('N pixels=', n_pixels)

        label_size_list.append({'label': label, 'size': n_pixels})

    # Sort the items by 'size' in descending order
    label_size_list = sorted(label_size_list, key=lambda x: x['size'], reverse=True)

    num_of_components = min(len(label_size_list), num_of_components)    
    print(f'num_of_components={num_of_components}')

    #
    print(f'taking the first {num_of_components} labels')
    label_size_list = label_size_list[:num_of_components]

    blobs = []
    for i, label_size in enumerate(label_size_list):
        label = label_size['label']
        size = label_size['size']

        img_th = sitk.Cast(rescale_intensity(sitk.Threshold(connected_component_image,   label-0.5, label+0.5, 0.0), out_min=0.0, out_max=1.0), sitk.sitkUInt8)

        blobs.append(img_th)
        
    if num_of_components == 1:
        return blobs[0]

    # combine the blobs    
    union_image = blobs[0]

    # Perform the union operation with each subsequent image
    for blob in blobs[1:]:
        union_image = sitk.Or(union_image, blob)
    
    return union_image
    
def process_extract_largest_connected_compoment(img_file, seg_file, args):

    print(f'process_extract_largest_connected_compoment(img_file={img_file}, seg_file={seg_file}, args={args})')

    # num_of_compoments
    num_of_compoments = int(args[0].split('=')[1])
    print(f'num_of_compoments={num_of_compoments}')

    print(f'reading seg_file...{seg_file}')
    seg = read_image(seg_file)
    print('extracting...')
    seg = extract_largest_connected_compoment(seg, num_of_components=num_of_compoments)
    print(f'saving result to ...{seg_file}')
    save_image(seg, seg_file)

def binary_dilate(binary_input_image, kernel_radius=1):

    structuring_element = sitk.sitkBall
    #kernel = sitk.BinaryBallStructuringElement(kernel_radius)

    # Apply BinaryDilate
    kernel_size = [kernel_radius] * binary_input_image.GetDimension() 
    dilated_image = sitk.BinaryDilate(binary_input_image, kernel_size, structuring_element)
    
    return dilated_image
 

def process_binary_dilate(img_file, seg_file, args):

    print(f'process_binary_dilate(img_file={img_file}, seg_file={seg_file}, args={args})')

    # num_of_compoments
    kernel_radius = int(args[0].split('=')[1])
    print(f'kernel_radius={kernel_radius}')

    print(f'reading seg_file...{seg_file}')
    seg = read_image(seg_file)
    print('extracting...')
    seg = binary_dilate(seg, kernel_radius=kernel_radius)
    print(f'saving result to ...{seg_file}')
    save_image(seg, seg_file)


def binary_erode(binary_input_image, kernel_radius=1):

    structuring_element = sitk.sitkBall
    #kernel = sitk.BinaryBallStructuringElement(kernel_radius)

    # Apply BinaryErode
    kernel_size = [kernel_radius] * binary_input_image.GetDimension() 
    dilated_image = sitk.BinaryErode(binary_input_image, kernel_size, structuring_element)
    
    return dilated_image
 

def process_binary_erode(img_file, seg_file, args):

    print(f'process_binary_erode(img_file={img_file}, seg_file={seg_file}, args={args})')

    # num_of_compoments
    kernel_radius = int(args[0].split('=')[1])
    print(f'kernel_radius={kernel_radius}')

    print(f'reading seg_file...{seg_file}')
    seg = read_image(seg_file)
    print('extracting...')
    seg = binary_erode(seg, kernel_radius=kernel_radius)
    print(f'saving result to ...{seg_file}')
    save_image(seg, seg_file)


def post_process(img_file, seg_file, method_list):
    for i, method_args in enumerate(method_list):
        print(f'method_args[{i}]={method_args}')
        method = method_args['method']
        args = method_args['args']
        if method == 'extract_largest_connected_compoment':
            process_extract_largest_connected_compoment(img_file, seg_file, args)
        elif method == 'binary_dilate':
            process_binary_dilate(img_file, seg_file, args)
        elif method == 'binary_erode':
            process_binary_erode(img_file, seg_file, args)

def calculate_dice_coefficient(image1, image2):

    # Ensure that the images are binary (containing only 0 and 1)
    binary_image1 = sitk.BinaryThreshold(image1, lowerThreshold=0.5, upperThreshold=1.5, insideValue=1, outsideValue=0)
    binary_image2 = sitk.BinaryThreshold(image2, lowerThreshold=0.5, upperThreshold=1.5, insideValue=1, outsideValue=0)

    # Calculate intersection and union
    intersection = sitk.And(binary_image1, binary_image2)
    union = sitk.Or(binary_image1, binary_image2)

    # Compute the Dice coefficient
    dice_coefficient = 2 * sitk.GetArrayFromImage(intersection).sum() / (
        sitk.GetArrayFromImage(binary_image1).sum() + sitk.GetArrayFromImage(binary_image2).sum()
    )
    return dice_coefficient

def dice_same_size(seg_file, str_file):

    seg = read_image(seg_file)
    str = read_image(str_file)

    np1 = sitk.GetArrayFromImage(seg)
    np2 = sitk.GetArrayFromImage(str)
    
    # flatten label and prediction tensors
    np1 = np1.reshape(-1)
    np2 = np2.reshape(-1)
    
    intersection = (np1 * np2).sum()
    np1_sum = np1.sum()
    np2_sum = np2.sum()
    dice = (2.*intersection) / (np1_sum + np2_sum)
    
    return dice


def dice(seg_file, str_file, grid_size, grid_spacing):

    seg = read_image(seg_file)
    str = read_image(str_file)

    # find the center of structure
    COM = find_COM_of_binary_image(str_file)

    # sample grid
    grid_size = np.array(grid_size).astype(np.uint32)
    grid_spacing = np.array(grid_spacing)
    grid_size_phy = grid_size * grid_spacing
    print(grid_size_phy)
    
    grid_org = COM - grid_size_phy/2.0

    # Define resampling parameters
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(grid_org)
    resampler.SetOutputDirection(str.GetDirection())
    resampler.SetSize(grid_size.tolist())
    resampler.SetOutputSpacing(grid_spacing.tolist())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    resampler.SetDefaultPixelValue(0)
    
    # sample
    seg_sampled = resampler.Execute(seg)
    str_sampled = resampler.Execute(str)

    #dir = '../../express_server/public/db/data/_deleteme'
    #save_image(seg_sampled, os.path.join(dir, "_seg.sampled.mha"))
    #save_image(str_sampled, os.path.join(dir, "_str.sampled.mha"))

    np1 = sitk.GetArrayFromImage(seg_sampled)
    np2 = sitk.GetArrayFromImage(str_sampled)
    
    # flatten label and prediction tensors
    np1 = np1.reshape(-1)
    np2 = np2.reshape(-1)
    
    intersection = (np1 * np2).sum()
    np1_sum = np1.sum()
    np2_sum = np2.sum()
    dice = (2.*intersection) / (np1_sum + np2_sum)
    
    return dice

def trace_contour(current_cont_index, contours, hierarchy, hole, return_list, depth=0, max_depth=500):
    
    # Adding a depth check to avoid infinite recursion
    if depth > max_depth:
        print(f"Maximum recursion depth of {max_depth} exceeded.")
        return

    if current_cont_index == -1:
        print("Base case reached, stopping recursion.")
        return

    # Existing logic to trace the contour
    print(f"Tracing contour {current_cont_index} at depth {depth}")
    
    contour = contours[current_cont_index]
    [next, previous, first_child, parent ] = hierarchy[current_cont_index]

    points = [ [ int(pos[0][0]), int(pos[0][1]) ] for pos in contour]
    
    return_list.append({'points': points, 'hole': hole})

    # process child if it has one
    if first_child != -1:
        trace_contour(first_child, contours, hierarchy, not hole, return_list, depth+1, max_depth)

    # process next
    if next != -1:
        trace_contour(next, contours, hierarchy, hole, return_list, depth+1, max_depth)

def resample_binary_image_at_image_grid(seg, img):
    return resample_img1_at_img2_grid(seg, img, defaultPixelValue=0, interpolator=sitk.sitkNearestNeighbor)

def resample_img1_at_img2_grid(img1, img2, defaultPixelValue=0, interpolator=sitk.sitkLinear):

    img_coord = get_image_coord_from_itkImage(img2)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(img_coord.spacing.tolist())
    resample.SetSize(img_coord.size.tolist())
    resample.SetOutputDirection(img_coord.direction)
    resample.SetOutputOrigin(img_coord.origin.tolist())
    resample.SetDefaultPixelValue(defaultPixelValue)
    resample.SetInterpolator(interpolator)

    return resample.Execute(img1)

def extract_binary_label_image(label_image_path, contour_number, output_path):
    """
    Extracts a binary mask where pixels == contour_number from the label image
    and saves it to the specified output path.

    Parameters:
        label_image_path (str): Path to the input label image (e.g., .mha).
        contour_number (int): Pixel value to extract (e.g., 1 = bladder).
        output_path (str): Path to save the binary mask image.
    """
    import os
    import SimpleITK as sitk
    import numpy as np

    if not os.path.exists(label_image_path):
        raise FileNotFoundError(f"Label image not found: {label_image_path}")

    print(f"Reading label image from: {label_image_path}")
    image = sitk.ReadImage(label_image_path)
    label_array = sitk.GetArrayFromImage(image)

    print(f"Extracting binary mask for label value: {contour_number}")
    binary_array = (label_array == contour_number).astype(np.uint8)

    binary_image = sitk.GetImageFromArray(binary_array)
    binary_image.CopyInformation(image)

    sitk.WriteImage(binary_image, output_path)
    print(f"Saved binary image to: {output_path}")


def composit_label_image_to_binary_images(label_image_path, label_map, out_dir, skip_if_output_exists=True):
    import SimpleITK as sitk
    import numpy as np
    import os

    if not os.path.exists(label_image_path):
        raise FileNotFoundError(f"Input label image not found: {label_image_path}")

    print(f"Splitting label image: {label_image_path}")
    print(f"Label map: {label_map}")

    output_filenames = []
    base_fname = os.path.basename(label_image_path)
    image = None
    label_array = None

    for label_name, label_value in label_map.items():
        if label_value == 0:
            continue  # skip background

        output_filename = f"{base_fname}.{label_value}.mha"
        output_path = os.path.join(out_dir, output_filename)

        if skip_if_output_exists and os.path.exists(output_path):
            print(f"Skipping existing file: {output_path}")
            output_filenames.append(output_filename)
            continue

        if image is None:
            image = sitk.ReadImage(label_image_path)
            label_array = sitk.GetArrayFromImage(image)

        binary_array = (label_array == label_value).astype(np.uint8)
        binary_image = sitk.GetImageFromArray(binary_array)
        binary_image.CopyInformation(image)

        sitk.WriteImage(binary_image, output_path)
        print(f"Saved binary mask for '{label_name}' to: {output_path}")
        output_filenames.append(output_filename)

    return output_filenames


def transform_contour_list(contour_list, H):
    contour_list_w = []
    for Z_contours in contour_list:
        slice = Z_contours['slice']
        contours = Z_contours['contours']
        contours_w = []
        for contour in contours:
            points = contour['points']
            hole = contour['hole']
            points_w = []
            for point in points:
                X = point[0]
                Y = point[1]
                Z = slice
                pt_I = np.array([X, Y, Z, 1.0])
                pt_w = H @ pt_I
                points_w.append([float(pt_w[0]), float(pt_w[1]), float(pt_w[2])])
            contours_w.append({'points': points_w, 'hole': hole})
        contour_list_w.append({'slice': slice, 'contours': contours_w})
    return contour_list_w

def binary_image_to_contour(mask):
    
    # Convert SimpleITK images to NumPy arrays
    mask_np = sitk.GetArrayFromImage(mask)

    contours_list_I = []  # List to store contours from all slices

    # for slice
    for z in range(mask_np.shape[0]):  # Iterate through each slice
        slice_2d = mask_np[z, :, :]  # Extract the 2D slice

        # Convert objects from value 1 to 255
        #_, binary_slice = cv2.threshold(slice_2d, 1, 255, cv2.THRESH_BINARY)

        slice_max = np.max(slice_2d)
        print(f'slice max={slice_max}')

        if slice_max == 0:
            continue

        if slice_max == 1:
            slice_2d = slice_2d * 255

        # Find contours in the binary slice
        contours, hierarchy = cv2.findContours(slice_2d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(f'num contours={len(contours)}')
        if len(contours) == 0:
            continue

        # The hierarchy array has the shape (1, N, 4), where N is the number of contours found. 
        # Each element of the array is a vector of four integers [next, previous, first_child, parent]:
        print(hierarchy.shape)
        
        # remove the first dim (which is always 1)
        hierarchy = hierarchy[0] 

        # check if any of the outer contour has child (hole)
        cont_num = 0
        hole = False
        return_list = []
        trace_contour(cont_num, contours, hierarchy, hole, return_list)

        contours_list_I.append({'slice':z, 'contours': return_list})

    return contours_list_I
    
def binary_image_to_contour_list_json_files(binary_image_path, base_image_path=None, out_dir=None, skip_if_output_exists=True):

    # input image
    if not os.path.exists(binary_image_path):
        raise Exception(f"Input file not found:{binary_image_path}")

    # out_dir
    if out_dir is None: # if out_dir is not given, use the binary image folder as the output folder
        out_dir =  os.path.dirname(binary_image_path)
    else:
        # create out_dir if not exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # output files
    out_path_header = os.path.join(out_dir, os.path.basename(binary_image_path))
    points_I_json = out_path_header +'.points_I.json'
    points_o_json = out_path_header +'.points_o.json'
    points_w_json = out_path_header +'.points_w.json'
    if os.path.exists(points_I_json) and os.path.exists(points_o_json) and os.path.exists(points_w_json) and skip_if_output_exists:
        print('all output files exists. so, skipping...')
        return [os.path.basename(points_I_json), os.path.basename(points_o_json),os.path.basename(points_w_json)]

    # binary segmentation image
    binary_image = read_image(binary_image_path)

    # if base_image is given, resample the seg at base image grid
    if base_image_path:
        base_image = read_image(base_image_path)

        # resample
        binary_image = resample_binary_image_at_image_grid(binary_image, base_image)

    # get contours in I
    contour_list_I = binary_image_to_contour(binary_image)

    # convert to w 
    img_coord = get_image_coord_from_itkImage(binary_image)
    contour_list_o = transform_contour_list(contour_list_I, img_coord.o_H_I())
    contour_list_w = transform_contour_list(contour_list_I, img_coord.w_H_I())

    # save contours
    dict_helper.save_to_json(contour_list_I, points_I_json)
    dict_helper.save_to_json(contour_list_o, points_o_json)
    dict_helper.save_to_json(contour_list_w, points_w_json)

    return [os.path.basename(points_I_json), os.path.basename(points_o_json),os.path.basename(points_w_json)]

def get_image_slice_indices_of_non_zero_pixel_values(image_path):

    image = read_image(image_path)

    # Convert the image to a numpy array for easier slicing
    image_array = sitk.GetArrayFromImage(image)

    # List to store indices of slices with the object
    slices_with_object = []
    slices_without_object = []

    # Iterate through each slice (assuming slices along the third dimension)
    for i in range(image_array.shape[0]):
        # Check if there are any non-zero pixels in the slice
        if image_array[i].any():
            slices_with_object.append(i)
        else:
            slices_without_object.append(i)

    #print(slices_with_object)

    return slices_with_object, slices_without_object

def draw_random_slice(slices_with_object, slices_without_object):
    
    N_obj = len(slices_with_object)
    N_noobj = len(slices_without_object)
    N = N_obj+N_noobj

    if N == 0:
        raise Exception("There is no slice to sample from!")
    
    # if there is no slice with the object, sample one index from the without list
    if N_obj == 0:
        obj_exists = False
        return random.choice(slices_without_object), obj_exists
    
    # if there is no slice without the object, sample one index from the with list
    if N_noobj == 0:
        obj_exists = True
        return random.choice(slices_with_object), obj_exists
        
    # Randomly choose between the two lists
    if random.choice([True, False]):
        # Draw from slices with object
        obj_exists = True
        return random.choice(slices_with_object), obj_exists
    else:
        # Draw from slices without object
        obj_exists = False
        return random.choice(slices_without_object), obj_exists

def extract_slice(image, slice_index):
    # Get the size of the input image
    size = list(image.GetSize())
    
    # Set the size of the slice along the third dimension to 1
    size[2] = 1
    
    # Define the start index for extraction
    start = [0, 0, slice_index]
    
    # Define the region to extract
    extraction_region = image[start[0]:start[0]+size[0], start[1]:start[1]+size[1], start[2]:start[2]+size[2]]
    
    # Use the Extract function to get the slice
    slice_image = sitk.Extract(image, size, start)
    
    return slice_image

if __name__ == '__main__':
    
    import json

    outputs_dir = '/gpfs/projects/KimGroup/data/mic-mkfz/predictions/Dataset015_CBCTBladderRectumBowel2/req_034/outputs'
    dataset_json = os.path.join(outputs_dir, 'dataset.json')
    label_image_path = os.path.join(outputs_dir, 'image_0.mha')

    # Read JSON file
    with open(dataset_json, 'r') as file:
        dataset = json.load(file)

    # Accessing data
    
    labels_map = dataset["labels"]

    binary_image_filenames = composit_label_image_to_binary_images(label_image_path, label_map=labels_map, out_dir=outputs_dir, skip_if_output_exists=True)

    for binary_image_filename in binary_image_filenames:
        binary_image_path = os.path.join(outputs_dir, binary_image_filename) 
        contour_filenames = binary_image_to_contour_list_json_files(binary_image_path)
        print(contour_filenames)


