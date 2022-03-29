#!/usr/bin/env python3
import sys
import getopt
import numpy as np
import pandas as pd
from skimage import io as skio
from scipy import ndimage
from sklearn.preprocessing import QuantileTransformer

#################################################
# BinConf
# 
class BinConf:
    def __init__(self,binsize):
        self.gene_binsize = binsize
        self.body_binsize = binsize
        self.body_scale = 1

    def geneBinsize(self):
        return self.gene_binsize

    def bodyBinConf(self):
        return self.body_binsize , self.body_scale

#################################################
# BorderDetect 
# 
class BorderDetect:
    def __init__(self,x,y):
        self.x = x 
        self.y = y

    def Border(self):
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        ymin = np.min(self.y)
        ymax = np.max(self.y)
        self.x = self.x -xmin + 5
        self.y = self.y -ymin + 5
        height = int(ymax-ymin+10)
        width = int(xmax-xmin+10)
        mask = np.zeros((height,width),dtype=int)
        mask[self.y,self.x] = 1
        # open and close to remove noise and fill holes
        mask = ndimage.binary_opening(mask).astype(int)
        mask = ndimage.binary_closing(mask).astype(int)
        
        for y in range(0,height):
            for x in range(2,width-1):
                if mask[y,x-2]==0 and mask[y,x-1]==0 and mask[y,x]==0 and mask[y,x+1]==1:
                    mask[y,x] = 255
            for x in range(width-3,1,-1):
                if mask[y,x+2]==0 and mask[y,x+1]==0 and mask[y,x]==0 and mask[y,x-1]==1:
                    mask[y,x] = 255
        for x in range(0,width):
            for y in range(2,height-1):
                if mask[y-2,x]==0 and mask[y-1,x]==0 and mask[y,x]==0 and mask[y+1,x]==1:
                    mask[y,x] = 255
            for y in range(height-3,1,-1):
                if mask[y+2,x]==0 and mask[y+1,x]==0 and mask[y,x]==0 and mask[y-1,x]==1:
                    mask[y,x] = 255
        mask[self.y,self.x] = 0
        (y_idx,x_idx) = np.nonzero(mask)
        y_idx = y_idx + ymin - 5
        x_idx = x_idx + xmin - 5
        return y_idx,x_idx

#################################################
# BodyInfo 
# 
class BodyInfo:
    def __init__(self,bin_border,bin_draw_scale):
        self.bin_border = bin_border
        self.bin_draw_scale = bin_draw_scale

    def loadAllPoints(self,xyz_txt):
        body=np.loadtxt(xyz_txt)
        bd = pd.DataFrame(columns=['x','y','z']);

        bd['x']= body[:,0]
        bd['x']= bd['x']/self.bin_border
        bd['x']= bd['x'].astype(int)

        bd['y']= body[:,1]
        bd['y']= bd['y']/self.bin_border
        bd['y']= bd['y'].astype(int)

        bd['z']= body[:,2]
        bd['z']= bd['z']/self.bin_border
        bd['z']= bd['z'].astype(int)
        self.body = bd

    #################################
    #  AP = x axis , ML = y axis
    #
    def calcAPML_border(self):              
        # agg all z coordinate
        bd = self.body.groupby(['x', 'y']).agg(z=('z','max')).reset_index()
        height = int(np.max(bd['y'])+10)
        width = int(np.max(bd['x'])+10)
        # get basic infos
        self.APML_W  = width * self.bin_draw_scale
        self.APML_H  = height * self.bin_draw_scale
        self.APML_points_num = len(bd)*self.bin_draw_scale*self.bin_draw_scale
        # get border dash points
        graph_matrix = np.zeros((height,width),dtype='uint8')
        graph_matrix[bd['y'],bd['x']]=1
        ( body_y, body_x ) = np.nonzero(graph_matrix)
        y_idx,x_idx = BorderDetect(body_x,body_y).Border()
        # save final border
        self.APML_x_idx = x_idx*self.bin_draw_scale
        self.APML_y_idx = y_idx*self.bin_draw_scale

    def getAPML_num_points(self):
        return self.APML_points_num

    def getAPML_WH(self):
        return self.APML_W,self.APML_H

    def getAPML_border(self):   
        return self.APML_x_idx , self.APML_y_idx           
    
    #################################
    #  AP = x axis , DV = y axis
    #
    def calcAPDV_border(self):              
        # agg all y coordinate
        bd = self.body.groupby(['x', 'z']).agg(y=('y','max')).reset_index()
        height = int(np.max(bd['z'])+10)
        width = int(np.max(bd['x'])+10)
        # get basic infos
        self.APDV_W  = width * self.bin_draw_scale
        self.APDV_H  = height * self.bin_draw_scale
        self.APDV_points_num = len(bd)*self.bin_draw_scale*self.bin_draw_scale
        # get border dash points
        graph_matrix = np.zeros((height,width),dtype='uint8')
        graph_matrix[bd['z'],bd['x']]=1
        ( body_y, body_x ) = np.nonzero(graph_matrix)
        y_idx,x_idx = BorderDetect(body_x,body_y).Border()
        # save final border
        self.APDV_x_idx = x_idx*self.bin_draw_scale
        self.APDV_y_idx = y_idx*self.bin_draw_scale

    def getAPDV_num_points(self):
        return self.APDV_points_num

    def getAPDV_WH(self):
        return self.APDV_W, self.APDV_H

    def getAPDV_border(self):   
        return self.APDV_x_idx , self.APDV_y_idx           

    #################################
    #  ML = x axis , DV = x axis
    #
    def calcMLDV_border(self):              
        # agg all y coordinate
        bd = self.body.groupby(['y', 'z']).agg(x=('x','max')).reset_index()
        height = int(np.max(bd['y'])+10)
        width = int(np.max(bd['z'])+10)
        # get basic infos
        self.MLDV_W  = width * self.bin_draw_scale
        self.MLDV_H  = height * self.bin_draw_scale
        self.MLDV_points_num = len(bd)*self.bin_draw_scale*self.bin_draw_scale
        # get border dash points
        graph_matrix = np.zeros((height,width),dtype='uint8')
        graph_matrix[bd['y'],bd['z']]=1
        ( body_y, body_x ) = np.nonzero(graph_matrix)
        y_idx,x_idx = BorderDetect(body_x,body_y).Border()
        # save final border
        self.MLDV_x_idx = x_idx*self.bin_draw_scale
        self.MLDV_y_idx = y_idx*self.bin_draw_scale

    def getMLDV_num_points(self):
        return self.MLDV_points_num

    def getMLDV_WH(self):
        return self.MLDV_W, self.MLDV_H

    def getMLDV_border(self):   
        return self.MLDV_x_idx , self.MLDV_y_idx           

class Gene3D:
    def __init__(self,binsize):
        self.binsize = binsize

    def loadExpr(self,gene_txt):
        a = np.loadtxt(gene_txt)
        if len(a) == 0 :
           self.valid = False   
           return
        show_data = pd.DataFrame(columns=['x','y','z','value'])
        self.valid = True
        show_data['x'] = a[:,0]
        show_data['y'] = a[:,1]
        show_data['z'] = a[:,2]
        show_data['value'] = a[:,3]
        self.gene_expr = show_data

    def getMIR_APML(self):
        show_data = self.gene_expr.copy()
        show_data['x'] = show_data['x']/self.binsize
        show_data['x'] = show_data['x'].astype(int)
        show_data['y'] = show_data['y']/self.binsize
        show_data['y'] = show_data['y'].astype(int)
        show_data = show_data.groupby(['x', 'y']).agg(value=('value', 'max')).reset_index()
        return show_data

    def getMIR_APDV(self):
        show_data = self.gene_expr.copy()
        show_data['x'] = show_data['x']/self.binsize
        show_data['x'] = show_data['x'].astype(int)
        show_data['z'] = show_data['z']/self.binsize
        show_data['z'] = show_data['z'].astype(int)
        show_data = show_data.groupby(['x', 'z']).agg(value=('value', 'max')).reset_index()
        return show_data

    def getMIR_MLDV(self):
        show_data = self.gene_expr.copy()
        show_data['y'] = show_data['y']/self.binsize
        show_data['y'] = show_data['y'].astype(int)
        show_data['z'] = show_data['z']/self.binsize
        show_data['z'] = show_data['z'].astype(int)
        show_data = show_data.groupby(['y', 'z']).agg(value=('value', 'max')).reset_index()
        return show_data
   
def GetBodyInfo(body_txt,binconf):
    body_binsize, body_scale = binconf.bodyBinConf()
    body_info = BodyInfo(body_binsize,body_scale)
    body_info.loadAllPoints(body_txt)
    return body_info

def GetBackground(view,body_info):
    if view == "APML" :
        body_info.calcAPML_border()
        W,H = body_info.getAPML_WH()
        draw_array = np.zeros((H,W,3),dtype='uint8')
        xids,yids = body_info.getAPML_border()
        # draw background
        draw_array[yids,xids,:] = 255
        # draw scale bar
        draw_array[[H-10,H-9,H-8],W-15:W-5,:]=255
        return draw_array
    elif view == "APDV" :
        body_info.calcAPDV_border()
        W,H = body_info.getAPDV_WH()
        draw_array = np.zeros((H,W,3),dtype='uint8')
        xids,yids = body_info.getAPDV_border()
        # draw background
        draw_array[yids,xids,:] = 255
        # draw scale bar
        draw_array[[H-10,H-9,H-8],W-15:W-5,:]=255
        return draw_array
        #return None
    elif view == "MLDV" :
        body_info.calcMLDV_border()
        W,H = body_info.getMLDV_WH()
        draw_array = np.zeros((H,W,3),dtype='uint8')
        xids,yids = body_info.getMLDV_border()
        # draw background
        draw_array[yids,xids,:] = 255
        # draw scale bar
        draw_array[[H-10,H-9,H-8],W-15:W-5,:]=255
        return draw_array

def GetGeneExpr(gene_txt,binconf):
    gene_expr = Gene3D(binconf.geneBinsize())
    gene_expr.loadExpr(gene_txt)
    return gene_expr

def FISH_scale(num_points, panel_expr):
    min_value = np.min(panel_expr)
    max_value = np.max(panel_expr)
    #print(f'min = {min_value}; max={max_value}',flush=True)
    num_light_panel=len(panel_expr)
    temp_list = np.zeros(num_points)
    temp_list[0:num_light_panel]=panel_expr
    qt = QuantileTransformer(output_distribution='uniform', random_state=0)
    temp_list_scale = qt.fit_transform(temp_list.reshape(-1, 1))
    temp_list_scale = temp_list_scale.reshape(-1)
    ret_data = temp_list_scale[0:num_light_panel]
    ret_data = ret_data*8
    ret_data = np.power((np.ones(len(ret_data))*2),ret_data);
    ret_data [ ret_data >255 ] = 255
    ret_data = ret_data.astype(int)
    return ret_data
    
def DrawSingleFISH_APML( body_info, expr, channel_id):
    W,H = body_info.getAPML_WH()
    draw_array = np.zeros((H,W,3),dtype='uint8')
    APML_expr = expr.getMIR_APML()
    draw_expr =  FISH_scale(body_info.getAPML_num_points(),APML_expr['value'])
    if channel_id == 0: 
        draw_array[APML_expr['y'],APML_expr['x'],0] = draw_expr
        draw_array[APML_expr['y'],APML_expr['x'],2] = draw_expr
    elif channel_id == 2: 
        draw_array[APML_expr['y'],APML_expr['x'],0] = draw_expr
        draw_array[APML_expr['y'],APML_expr['x'],1] = draw_expr
    else:
        draw_array[APML_expr['y'],APML_expr['x'],channel_id] = draw_expr
    return draw_array 

def DrawSingleFISH_APDV(body_info, expr, channel_id):
    W,H = body_info.getAPDV_WH()
    draw_array = np.zeros((H,W,3),dtype='uint8')
    APDV_expr = expr.getMIR_APDV()
    draw_expr = FISH_scale(body_info.getAPDV_num_points(),APDV_expr['value'])
    if channel_id == 0: 
        draw_array[APDV_expr['z'],APDV_expr['x'],0] = draw_expr
        draw_array[APDV_expr['z'],APDV_expr['x'],2] = draw_expr
    elif channel_id == 2: 
        draw_array[APDV_expr['z'],APDV_expr['x'],0] = draw_expr
        draw_array[APDV_expr['z'],APDV_expr['x'],1] = draw_expr
    else:
        draw_array[APDV_expr['z'],APDV_expr['x'],channel_id] = draw_expr
    return draw_array 

def DrawSingleFISH_DVML(body_info, expr, channel_id):
    W,H = body_info.getMLDV_WH()
    draw_array = np.zeros((H,W,3),dtype='uint8')
    MLDV_expr = expr.getMIR_MLDV()
    draw_expr = FISH_scale(body_info.getMLDV_num_points(),MLDV_expr['value'])
    if channel_id == 0: 
        draw_array[MLDV_expr['y'],MLDV_expr['z'],0] = draw_expr
        draw_array[MLDV_expr['y'],MLDV_expr['z'],2] = draw_expr
    elif channel_id == 2: 
        draw_array[MLDV_expr['y'],MLDV_expr['z'],0] = draw_expr
        draw_array[MLDV_expr['y'],MLDV_expr['z'],1] = draw_expr
    else:
        draw_array[MLDV_expr['y'],MLDV_expr['z'],channel_id] = draw_expr
    return draw_array 

def DrawSingleFISH(view, body_info, gene_expr, channel_id):
    if view == "APML" :
        return DrawSingleFISH_APML(body_info, gene_expr, channel_id)
    elif view == "APDV":
        return DrawSingleFISH_APDV(body_info, gene_expr, channel_id)
    elif view == "MLDV":
        return DrawSingleFISH_DVML(body_info, gene_expr, channel_id)

############################################################################
# Usage
#
def FISH_like_usage():
    print("""
Usage : FISH_3c.py -i <individual.txt> \\
                   -o <output prefix>  \\
                   -g [gene.txt that draw in Green channel] \\
                   -m [gene.txt that draw in Magenta channel]   \\
                   -y [gene.txt that draw in Yellow channel]  \\
                   --view [default APML APML/APDV/MLDV] \\
                   --xmin [default 0] \\
                   --ymin [default 0] \\
                   --zmin [default 0] \\
                   --xmax [default 1000] \\
                   --ymax [default 1000] \\
                   --zmax [default 1000] \\
                   --binsize [default 10] 

Example :
    python3 FISH_3c.py -i WT.txt -o WT_wnt1 -g wnt1.txt 

Notice :
    please use at least one of -g, -m, or -y.

""")

def FISH_like_main(argv:[]):
    ###############################################################################
    # Default values
    indv = ''
    prefix = ''
    g_gene = ''
    r_gene = ''
    b_gene = ''
    view = 'APML'
    xmin = ymin = zmin = 0
    xmax = ymax = zmax = 1000
    binsize = 10

    ###############################################################################
    # Parse the arguments
    try:
        opts, args = getopt.getopt(argv,"hi:o:m:g:y:",["help","view=","xmin=","ymin=","zmin=","xmax=","ymax=","zmax=","binsize="])
    except getopt.GetoptError:
        FISH_like_usage()
        sys.exit(2)
    for opt, arg in opts:
        print(f' {opt} {arg} ')
        if opt in ('-h' ,'--help'):
            FISH_like_usage()
            sys.exit(0)
        elif opt in ("-i"):
            indv = arg
        elif opt in ("-o"):
            prefix = arg
        elif opt == "-m" :
            r_gene = arg
        elif opt == "-b" :
            b_gene = arg
        elif opt == "-y" :
            g_gene = arg
        elif opt == "--xmin":
            xmin = int(arg)
        elif opt == "--ymin":
            ymin = int(arg)
        elif opt == "--zmin":
            zmin = int(arg)
        elif opt == "--xmax":
            xmax = int(arg)
        elif opt == "--ymax":
            ymax = int(arg)
        elif opt == "--zmax":
            zmax = int(arg)
        elif opt == "--view":
            view = arg
        elif opt == "--binsize":
            binsize = int(opt)
    
    ###############################################################################
    # Sanity check
    if indv == "" or ( g_gene == "" and r_gene == "" and b_gene == "" ) or prefix == "":
        FISH_like_usage()
        sys.exit(3)
    
    binconf = BinConf(binsize)
    print(f"the drawing view : {view}")
    ###############################################################################
    # Load the body points 
    print('Loading body now ...',flush=True)
    body_info = GetBodyInfo(indv,binconf)
    ###############################################################################
    # Load the gene expr points and draw
    print('Loading gene expression now ...',flush=True)
    r_gene_expr = g_gene_expr = b_gene_expr = None 
    channel = 0
    if g_gene != "" :
        g_gene_expr = GetGeneExpr(g_gene,binconf) 
        if g_gene_expr.valid :
            channel = channel + 1
        else :
            g_gene_expr = None
    if r_gene != "" :
        r_gene_expr = GetGeneExpr(r_gene,binconf) 
        if r_gene_expr.valid :
            channel = channel + 1
        else :
            r_gene_expr = None
    if b_gene != "" :
        b_gene_expr = GetGeneExpr(b_gene,binconf) 
        if b_gene_expr.valid :
            channel = channel + 1
        else :
            b_gene_expr = None

    ###############################################################################
    # get sample border
    background = GetBackground(view,body_info)
    ###############################################################################
    # Draw single FISH
    r_image = b_image = g_image = None
    if r_gene_expr != None :
        r_image = DrawSingleFISH(view,body_info,r_gene_expr,0)
        if channel == 1 :
            print('Draw red-channel-FISH now ...',flush=True)
            draw_image = background + r_image
            skio.imsave(f'{prefix}.magenta.MIR.tif',draw_image)
    if g_gene_expr != None :
        g_image = DrawSingleFISH(view,body_info,g_gene_expr,1)
        if channel == 1 :
            print('Draw green-channel-FISH now ...',flush=True)
            draw_image = background + g_image 
            skio.imsave(f'{prefix}.green.MIR.tif',draw_image)
    if b_gene_expr != None :
        b_image = DrawSingleFISH(view,body_info,b_gene_expr,2)
        if channel == 1 :
            print('Draw blue-channel-FISH now ...',flush=True)
            draw_image = background + b_image 
            skio.imsave(f'{prefix}.yellow.MIR.tif',draw_image)

    ###############################################################################
    # Draw multi-FISH
    if channel > 1:
        print('Draw multi-FISH now ...',flush=True)
        if r_gene_expr != None :
            draw_image = background + r_image
        if g_gene_expr != None :
            draw_image = draw_image + g_image
        if b_gene_expr != None :
            draw_image = draw_image + b_image
        skio.imsave(f'{prefix}.multi.MIR.tif',draw_image)
    
    ###############################################################################
    # Done
    print('__ALL DONE__',flush=True)

if __name__ == "__main__":
    FISH_like_main(sys.argv[1:])
