"""Computation of pyPALC, without Bokeh.

The code uses descriptive names. The table relates them to the symbols of
the PALC papers (Straube et al. 2018, JAES 66(1/2); Hölter et al. 2021);
the Development page of the documentation lists the former names of the
code.

==========================================  ===========  ======================================================================
Name                                        Paper        Meaning
==========================================  ===========  ======================================================================
:class:`~.venue.AudienceLines`                           drawn audience lines, discretized
:class:`~.venue.PalcVenue`                               venue in the format of the PALC algorithm
:class:`~.palc.PalcConfig`                               configuration of the PALC algorithm
:class:`~.sound_field.SoundFieldConfig`                  configuration of the sound field prediction
:class:`~.palc.ArrayGeometry`                            geometry of an array (angles, rays, hit points)
:class:`~.sound_field.ArrayEvaluation`                   sound field, homogeneity and histogram of an array
:class:`~.target_slope.TargetSlopeState`                 target slope, or an SPL over distance
:class:`~.palc.RayPlotData`                              rays and cabinets for plotting
``PlotRanges`` (:mod:`pypalc.gui.ranges`)                ranges of the figures
``cabinet_count``                           N            number of cabinets
``cabinet_height``                          Λ_y          height of a cabinet in m
``cabinet_gap``                             Λ_gap        gap between two cabinets in m
``top_x``, ``top_y``                        x_H, y_H     front top edge of the uppermost cabinet in m
``aperture_angles``                         ψ_n          aperture (splay) angles in rad
``tilt_angles``                             γ_n          tilt angles in rad
``tilt_angles_deg``                                      tilt angles in degree
``inter_cabinet_angles_deg``                             angles between adjacent cabinets in degree
``discrete_tilt_angles``                                 allowed inter cabinet angles in rad
``use_discrete_angles``                                  if the discrete angles are used
``palc_variant``                                         'PALC 1', 'PALC 2' or 'PALC 3'
``segments``, ``audience_segments``         PAL          venue segments with and without non-audience lines
``non_audience_lines``                                   suggested lines in gaps of the venue
``points_x``, ``points_y``                               discrete venue points in m
``line_end_indices``                                     index of the last point of each line
``cabinet_centers_x``                                    centers of the cabinets in m (x)
``cabinet_centers_y``                                    centers of the cabinets in m (y)
``hit_points``                                           points where the center ray of a cabinet hits the venue
``hit_points_start``                                     points where the upper ray of a cabinet hits the venue
``hit_points_stop``                                      points where the lower ray of a cabinet hits the venue
``spl_db``                                               sound pressure level in dB
``frequencies``                                          frequency bins in Hz
``homogeneity_db``                          H(f)         homogeneity in dB
``homogeneity_low_db``                                   lower reference line of the homogeneity figure
``homogeneity_high_db``                                  upper reference line of the homogeneity figure
``spl_over_distance``                                    SPL over the distance from the array
``evaluation``                                           :class:`~.sound_field.ArrayEvaluation` of the optimized array
``evaluation_reference``                                 :class:`~.sound_field.ArrayEvaluation` of the reference array
``optimized_array``                                      :class:`~.palc.ArrayGeometry` of the optimized array
``reference_array``                                      :class:`~.palc.ArrayGeometry` of the reference array
``drawn_lines``                                          :class:`~.venue.AudienceLines` as drawn
``palc_venue``                                           :class:`~.venue.PalcVenue` made from the drawn lines
``weighting_nu``                            ν            parameter of linear and logarithmic weighting
==========================================  ===========  ======================================================================
"""
