#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

Everything that is placed on the 'Help' Tab can be written or found in this
module.
"""

from bokeh.layouts import row
from bokeh.models.widgets import Div, Panel, Tabs

def help_text():
    """
    Strings in html-format that are placed in Div widget in Rows to be displayed
    in the Tabs of the 'Help' page.

    Returns
    -------
    help_tabs : obj
        Tabs widget that contains all data of 'Help' page.

    """
    lspace = '<br><p></p>'
    llspace = '<br><p></p><br>'
    empty_head = '<b> </b>'
    ########################## Text for page 0 (general) ######################
    page0_info1 = '<b> General Information </b>'
    page0_general1 = 'This is a <i> Loudspeaker Array Curving Optimization Software</i>.'
    page0_authors1 = 'Programmed and developed by: <br> <b><a href="https://www.ak.tu-berlin.de/menue/team/wissenschaftliche_mitarbeiter/arne_hoelter/" target="_blank">Arne Hölter</a></b> and <b>Florian Straube</b>'#<b><a href="https://www.ak.tu-berlin.de/menue/team/wissenschaftliche_mitarbeiter/dr_florian_straube/" target="_blank">Florian Straube</a></b>'
    page0_authors2 = 'Correspondence should be addressed to: <br> <a href="mailto:hoelter@campus.tu-berlin.de?subject=PALC-App">hoelter@campus.tu-berlin.de</a>'
    page0_general2 = 'Based on the following publications:'
    page0_paper1 = '<b>[1] Straube, F.; Schultz, F.; Bonillo, D.A.; Weinzierl, S. (2018): “An Analytical Approach for Optimizing the Curving of Line Source Arrays.” In: <i>J. Audio Eng. Soc.</i> 66(1/2), 4–20.</b>' #&nbsp
    page0_paper2 = '<b>[2] Hölter, A.; Straube, F.; Schultz, F.; Weinzierl, S. (2020): “Eine Web-Applikation zu Optimierung der Krümmung von Line Source Arrays.” In: <i>Fortschritte der Akustik: Tagungsband d. 46. DAGA</i>, Hannover.</b>'
    page0_paper3 = '<b>[3] Hölter, A.; Straube, F.; Schultz, F.; Weinzierl, S. (2021): “Enhanced Polygonal Audience Line Curving for Line Source Arrays.” In: <i>Proc. of the 149th Audio Eng. Soc. Conv.</i>, Online.</b>'
    page0_paper4 = '<b>[4] Hölter, A.; Schultz, F.; Straube F.; Weinzierl, S. (2021): "SPL-basierte Optimierung der Krümmung von Line-Source-Arrays mit PALC." In: <i>Fortschritte der Akustik: Tagungsband d. 47. DAGA</i>, Wien.</b>'
    page0_col0 = Div(text = page0_info1 + llspace + page0_general1 + lspace + page0_authors1 + lspace + page0_authors2, width=350)
    page0_col1 = Div(text = empty_head + llspace + page0_general2 + lspace + page0_paper1 + lspace + page0_paper2 + lspace + page0_paper3 + lspace + page0_paper4, width=350)
    
    ########################## Text for page 1 ################################
    page1_info1 = "<b>I. Page - Create the Venue!</b>"
    page1_info2 = "It is only possible to create straight lines, see below for more information."
    page1_1 = "1. Enter the <b>start position</b> of an audience line on the x-axis in meter."
    page1_2 = "2. Enter the <b>end position</b> of an audience line on the x-axis in meter."
    page1_3 = "3. Enter the <b>discretization</b> of an audience line, i.e., the number of points per meter. Only integer values are allowed. Note: The maximum of discretized points is 1000, e.g., a discretization of 10 points per meter would allow around 100 meter of audience lines."
    page1_4 = "4. Enter the <b>start position</b> of an audience line on the y-axis in meter."
    page1_5 = "5. Enter the <b>end position</b> of an audience line on the y-axis in meter."
    page1_6 = "6. Click on the button to <b>save</b> the current audience line. It will be displayed in the right figure. If <b>tapped</b> on a line, the line will be created in front of the tapped line."
    page1_7 = "7. Click on the button to <b>remove</b> the last audience line that was created. It is also possible after computation. If <b>tapped</b> on a line, the selected line will be deleted."
    page1_8 = "8. Click on the button to <b>get</b> a PALC compatible venue proposal, i.e., a continuous polygonal audience line (PAL) is created based on the various drawn audience lines. The gaps between the different audience lines are processed."
    page1_col0 = Div(text = page1_info1 + llspace + page1_info2 + lspace + page1_1 + lspace + page1_2 + lspace + page1_3, width=350)
    page1_col1 = Div(text = empty_head + llspace + page1_4 + lspace + page1_5 + lspace + page1_6 + lspace + page1_7 + lspace + page1_8, width=350)

    ########################## Text for page 2 ################################
    page2_info = "<b>II. Configure the Loudspeakers!</b>"
    page2_1 = "9. The <b>PALC aperture angle</b> of the loudspeakers in degree is the start value for the iterative PALC calculation. In some cases it can be useful to set the start value lower than the default value, e.g., if many loudspeakers shall reinforce a small venue. In most of the cases the default value can be maintained."
    page2_2 = "10. The <b>loudspeaker height</b> in meter. The front grille's height of the deployed loudspeaker cabinets."
    page2_3 = "11a. Choose the <b>directivity</b> of the loudspeakers. Either modeled or measured data. Modeled: the directivities <i>circular piston</i>, <i>line piston</i> or a <i>combination of a circular and a line piston</i> are possible. For the latter the circular piston is used for the low frequencies and the line piston is used for the high frequencies. The crossover is modeled with a 4th-order Linkwitz-Riley filter at 1.5 kHz."
    page2_4 = "11b. <b>Upload measured loudspeaker data</b>: Only .csv-files are accepted. The first entry must be a zero as a placeholder. The first line must contain the frequency bins in Hz. The first row must contain the angles in degree. The rest of the table must contain the complex directivity values in cartesian representation (amplitude and phase). The imaginary unit is 'i' or 'j'."
    page2_col0 = Div(text = page2_info + llspace + page2_1 + lspace + page2_2, width=350)
    page2_col1 = Div(text = empty_head + llspace + page2_3 + lspace + page2_4, width=350)

    ########################## Text for page 3 ################################
    page3_info = "<b>III. Configure the Loudspeaker Array!</b>"
    page3_1 = "12. Enter the <b>number</b> of loudspeaker cabinets which the array is built of."
    page3_2 = "13. Enter the <b>x-coordinate</b> of the top front edge of the array's top loudspeaker cabinet in meter."
    page3_3 = "14. Enter the <b>y-coordinate</b> of the top front edge of the array's top loudspeaker cabinet in meter."
    page3_4 = "15. Enter the <b>gap</b> between the loudspeaker cabinets within the array in meter."
    page3_5 = "16. Choose if the calculation is performed with a <b>set of discrete tilt angles</b> in degree."
    page3_6 = "17. Enabled if (15.) is set to 'Yes'. Enter a <b>set of discrete tilt angles</b> that will be used for the line array curving. The value separator is ',' and the decimal separator is '.', e.g., [0,1,2.5,3,5]. This means the possible inter cabinet tilt angles are 0 deg, 1 deg, 2.5 deg, 3 deg and 5 deg."
    page3_col0 = Div(text = page3_info + llspace + page3_1 + lspace + page3_2 + lspace + page3_3, width=350)  
    page3_col1 = Div(text = empty_head + llspace + page3_4 + lspace + page3_5 + lspace + page3_6, width=350)
    
    ########################## Text for page 4 ################################
    page4_info = "<b>IV. Algorithm Options and Calculation!</b>"
    page4_1 = "18. Choose the <b>PALC-algorithm</b>. <i>PALC1</i>: the PALC aperture angles of all loudspeaker cabinets are set to be constant. <i>PALC2</i>: the product of the PALC aperture angles of a loudspeaker cabinet and the distance of the acoustic center of the respective loudspeaker cabinet to the audience line is set to be constant. <i>PALC3</i> is similar to PALC2 but it takes the tangent of the PALC aperture angle instead of the approximated angle itself."
    page4_2 = "19. <b>Gap handling approach</b>. <i>Without</i>: No additional lines are inserted in the gaps between the audience lines. <i>Hard Margin</i>: The gaps are not handled as audience lines and the calculation runs separately for each continuous audience line. The combination with target slope weighting optimization is not possible. <i>Soft Margin</i>: Non audience lines are inserted in the gaps between the audience lines. They are treated with a lower weighting in the calculation."
    page4_3 = "20. The <b>strength of the soft margin approach</b> is enabled when choosing soft margin in (19.). Raise the slider to use lower weighting for gaps or reduce the slider to converge weighting of the gaps to the audience lines."
    page4_4 = "21. <b>Weighting</b>: Choose a weighting in order to affect the SPL loss over distance: linear spacing, logarithmic spacing or a target slope. <br> If linear or logarithmic spacing is chosen, choose the strength of the weighting factors at the slider below. <br> If Target Slope is chosen, the SPL loss over distance will be optimized regarding the target slope. Choose the number of hinges (0, 1 or 2). The combination with the hard margin gap handling approach is not possible."
    page4_5 = "22. Adjust the weighting parameter by clicking on: <b>+</b> to increase the SPL loss over distance, <b>-</b> to decrease the SPL loss over distance."
    page4_6 = "23. Choose the <b>tolerance</b> as criterion for the iterative algorithm to abort. If no convergence is reached in a calculation, it should always be the first step to increase the abort criterion."
    page4_7 = "24. If the uppermost loudspeaker cabinet within the array is supposed to be mounted with a specific, <b>fixed</b> tilt angle, change to 'Yes'."
    page4_8 = "25. If (24.) is set to 'Yes': Choose the <b>fixed tilt angle</b> of the uppermost loudspeaker cabinet. Note: It is only possible to change the angle in defined range depending on LSA and venue geometry."
    page4_9 = "26. <b>Run calculation</b>: Start the computation of the PALC algorithm. <br> <b>Note</b>, if no convergence is reached, decrease initial aperture angle in (9.) or increase the tolerance (23.)."
    page4_10 = "27. <b>Delete the result visualization</b>: Removes the visualization of the results. This is automatically executed if clicking on '1. PAL' in the menu."
    page4_col0 = Div(text = page4_info + llspace + page4_1 + lspace + page4_2, width=350)
    page4_col1 = Div(text = empty_head + llspace + page4_3 + lspace + page4_4 + lspace + page4_5, width=350)
    page4_col2 = Div(text = empty_head + llspace + page4_6 + lspace + page4_7 + lspace + page4_8 + lspace + page4_9 + lspace + page4_10, width=350)
    
    ########################## Text for page 5 ################################
    page5_info = "<b>V. Visualization of the Sound Field Prediction!</b>"
    page5_1 = "28. <b>Venue slice</b> with discrete audience line positions in the color of the corresponding SPL predictions in (29.) and (30.). Click on a specific discrete position to see the corresponding SPL prediction in (29.) and (30.)."
    page5_2 = "29. <b>SPL frequency responses</b> at all audience positions for the loudspeaker array using the <i>PALC</i> optimized tilt angles."
    page5_3 = "30. <b>SPL frequency responses</b> at all audience positions for the <i>reference loudspeaker array</i>."
    page5_4 = "31. <b>Homogeneity</b> over frequency at all audience positions for the loudspeaker array using the <i>PALC</i> optimized tilt angles in blue and for the <i>reference loudspeaker array</i> in red."
    page5_5 = "32. <b>Bar chart</b> of the SPL values at the audience positions for the loudspeaker array using the <i>PALC</i> optimized tilt angles."
    page5_6 = "33. <b>Bar chart</b> of the SPL values at the audience positions for the <i>reference loudspeaker array</i>."
    page5_7 = "34. <b>Beam plot</b> visualizing the SPL values in the arrays' vertical radiation plane."
    page5_col0 = Div(text = page5_info + llspace + page5_1 + lspace + page5_2 + lspace + page5_3, width=350)
    page5_col1 = Div(text = empty_head + llspace + page5_4 + lspace + page5_5 + lspace + page5_6 + lspace + page5_7, width=350)
    
    ##### Set up the Text Div Widgets Panels as Rows and Panels in Tabs #######
    help_tab0  = Panel(child=row([page0_col0, page0_col1]), title='General Information')
    help_tab1  = Panel(child=row([page1_col0, page1_col1]), title='Venue Slice')
    help_tab2  = Panel(child=row([page2_col0, page2_col1]), title='Loudspeaker')
    help_tab3  = Panel(child=row([page3_col0, page3_col1]), title='Array')
    help_tab4  = Panel(child=row([page4_col0, page4_col1, page4_col2]), title='Algorithm and Calculation')
    help_tab5  = Panel(child=row([page5_col0, page5_col1]), title='Visualization of the Sound Field Prediction')
    # Put the Panels in the Tabs
    help_tabs  = Tabs(tabs=[help_tab0, help_tab1, help_tab2, help_tab3, help_tab4, help_tab5])
   
    return help_tabs
