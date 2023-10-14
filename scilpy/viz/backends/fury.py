# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
from fury import window
from fury.colormap import distinguishable_colormap
from fury.utils import numpy_to_vtk_colors

from scilpy.utils.util import get_axis_index


class CamParams(Enum):
    """
    Enum containing camera parameters
    """
    VIEW_POS = 'view_position'
    VIEW_CENTER = 'view_center'
    VIEW_UP = 'up_vector'
    ZOOM_FACTOR = 'zoom_factor'


def initialize_camera(orientation, slice_index, volume_shape):
    """
    Initialize a camera for a given orientation.

    Parameters
    ----------
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.

    Returns
    -------
    camera : dict
        Dictionnary containing camera information.
    """
    camera = {}
    # heuristic for setting the camera position at a distance
    # proportional to the scale of the scene
    eye_distance = max(volume_shape)
    ax_idx = get_axis_index(orientation)

    if slice_index is None:
        slice_index = volume_shape[ax_idx] // 2

    view_pos_sign = [-1.0, 1.0, -1.0]
    camera[CamParams.VIEW_POS] = 0.5 * (np.array(volume_shape) - 1.0)
    camera[CamParams.VIEW_POS][ax_idx] = view_pos_sign[ax_idx] * eye_distance

    camera[CamParams.VIEW_CENTER] = 0.5 * (np.array(volume_shape) - 1.0)
    camera[CamParams.VIEW_CENTER][ax_idx] = slice_index

    camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
    if ax_idx == 2:
        camera[CamParams.VIEW_UP] = np.array([0.0, 1.0, 0.0])

    camera[CamParams.ZOOM_FACTOR] = 2.0 / \
        min(np.delete(volume_shape, ax_idx, 0))

    return camera


def set_display_extent(slicer_actor, orientation, volume_shape, slice_index):
    """
    Set the display extent for a fury actor in ``orientation``.

    Parameters
    ----------
    slicer_actor : actor
        Slicer actor from Fury
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    volume_shape : tuple
        Shape of the sliced volume.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    """

    ax_idx = get_axis_index(orientation)
    extents = np.vstack(([0., 0., 0.], volume_shape)).T.flatten()

    if slice_index is None:
        slice_index = volume_shape[ax_idx] // 2

    extents[2 * ax_idx:2 * ax_idx + 2] = slice_index
    slicer_actor.display_extent(*extents)


def create_scene(actors, orientation, slice_index,
                 volume_shape, bg_color=(0, 0, 0)):
    """
    Create a 3D scene containing actors fitting inside a grid. The camera is
    placed based on the orientation supplied by the user. The projection mode
    is parallel.

    Parameters
    ----------
    actors : list of actor
        Ensemble of actors from Fury
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.
    bg_color: tuple, optional
        Background color expressed as RGB triplet in the range [0, 1].

    Returns
    -------
    scene : window.Scene()
        Object from Fury containing the 3D scene.
    """
    # Configure camera
    camera = initialize_camera(orientation, slice_index, volume_shape)

    scene = window.Scene()
    scene.background(bg_color)
    scene.projection('parallel')
    scene.set_camera(position=camera[CamParams.VIEW_POS],
                     focal_point=camera[CamParams.VIEW_CENTER],
                     view_up=camera[CamParams.VIEW_UP])
    scene.zoom(camera[CamParams.ZOOM_FACTOR])

    # Add actors to the scene
    for curr_actor in actors:
        scene.add(curr_actor)

    return scene


def create_interactive_window(scene, window_size, interactor, title="Viewer", 
                              open_window=True):
    """
    Create a 3D window with the content of scene, equiped with an interactor.

    Parameters
    ----------
    scene : window.Scene()
        Object from Fury containing the 3D scene.
    window_size : tuple (width, height)
        The dimensions for the vtk window.
    interactor : str
        Specify interactor mode for vtk window. Choices are image or trackball.
    title : str, optional
        Title of the scene. Defaults to Viewer.
    open_window : bool, optional
        When true, initializes the interactor and opens the window 
        (This suspends the current thread).
    """
    showm = window.ShowManager(scene, title=title,
                               size=window_size,
                               reset_camera=False,
                               interactor_style=interactor)

    if open_window:
        showm.initialize()
        showm.start()

    return showm


def snapshot_scenes(scenes, window_size):
    return [window.snapshot(scene, size=window_size) for scene in scenes]