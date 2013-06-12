#include <highgui.h>
#include <iostream>
#include "maxflow.hpp"

using namespace std;
using namespace cv;

// class of the pixel in GrabCut algorithm
enum
{
    BG = 0,  //   background
    FG = 1,  //   foreground
    PBG = 2,  //   maybe background
    PFG = 3	   //   maybe foreground
};

//for each GMM model
class GMM
{
public:
    
    static const int modelNumInGMM = 5;
    
    GMM( Mat& model );
    
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    
    //which model does a pixel belong to
    int whichModel( const Vec3d color ) const;
    //xian pan duan color (pixel) shuyu nage ci, ranhou addsample dao ci
    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();
    
private:
    void calcInverseCovAndDet( int ci );
    
    Mat model;
    double* modelWeight;
    double* average;
    double* covariance;
    
    double inverseCovariance[modelNumInGMM][3][3];
    double covarianceDet[modelNumInGMM];
    double sums[modelNumInGMM][3];
    double prods[modelNumInGMM][3][3];
    int sampleCounts[modelNumInGMM];
    int totalSampleCount;
};

// (XXXXXXX)MAT store weight, average and cov for each model in this GMM
GMM::GMM( Mat& model )
{
    //the size for one model in GMM
    int modelSize = 3/*average*/ + 9/*covariance*/ + 1/*model weight*/;
    //if model is empty, create a GMM, one GMM has 5 model, one model is modelsize
    
    if( model.empty() )
    {
        model.create( 1, modelSize*modelNumInGMM, CV_64FC1 );
        model.setTo(0);
    }
    //mat size = 1 line and modelSize*componentsCount elements
    else if( (model.type() != CV_64FC1) || (model.rows != 1) || (model.cols != modelSize*modelNumInGMM) )
        CV_Error( CV_StsBadArg, "model type or size error" );
    
    
    //decide there to store these imformation (average, covariance and coef) for each model in the GMM
    modelWeight = model.ptr<double>(0);
    average = modelWeight + modelNumInGMM;
    covariance = average + 3*modelNumInGMM;
    
    //for each model
    for( int ci = 0; ci < modelNumInGMM; ci++ )
        if( modelWeight[ci] > 0 )
            calcInverseCovAndDet(ci);
}

//calculate the pro for the pixel belongs to this GMM
//pro = sum(modelweight*modelPro)
double GMM::operator()( const Vec3d color ) const
{
    double pro = 0;
    for( int ci = 0; ci < modelNumInGMM; ci++ )
        pro += modelWeight[ci] * (*this)(ci, color );
    return pro;
}

//calculate the pro for the pixel belongs to ci in this GMM 
double GMM::operator()( int ci, const Vec3d color ) const
{
    double pro = 0;
    if( modelWeight[ci] > 0 )
    {
       
        Vec3d diff;
        double* m = average + 3*ci;
        
        diff[0] = color[0] - m[0];
        diff[1] = color[1] - m[1];
        diff[2] = color[2] - m[2];
        
        double mult = diff[0]*(diff[0]*inverseCovariance[ci][0][0] + diff[1]*inverseCovariance[ci][1][0] + diff[2]*inverseCovariance[ci][2][0])
        + diff[1]*(diff[0]*inverseCovariance[ci][0][1] + diff[1]*inverseCovariance[ci][1][1] + diff[2]*inverseCovariance[ci][2][1])
        + diff[2]*(diff[0]*inverseCovariance[ci][0][2] + diff[1]*inverseCovariance[ci][1][2] + diff[2]*inverseCovariance[ci][2][2]);
        
        pro = 1.0f/sqrt(covarianceDet[ci]) * exp(-0.5f*mult);
    }
    return pro;
}

//decide which model the pixel belongs to based on the value od operation()
int GMM::whichModel( const Vec3d color ) const
{
    int maxModel = 0;
    double max = 0;
    
    for( int ci = 0; ci < modelNumInGMM; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            maxModel = ci;
            max = p;
        }
    }
    return maxModel;
}

void GMM::initLearning()
{
    //for each model in this GMM
    for( int ci = 0; ci < modelNumInGMM; ci++)
    {
        //sums[modelNumInGMM][3]   for average
        sums[ci][0] = 0;
        sums[ci][1] = 0;
        sums[ci][2] = 0;
        
        //prods[modelNumInGMM][3][3]  for cov
        prods[ci][0][0] = 0;
        prods[ci][0][1] = 0;
        prods[ci][0][2] = 0;
        prods[ci][1][0] = 0;
        prods[ci][1][1] = 0;
        prods[ci][1][2] = 0;
        prods[ci][2][0] = 0;
        prods[ci][2][1] = 0;
        prods[ci][2][2] = 0;
        
        //sampleCounts[modelNumInGMM] for modelWeight
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}
//set value for ci model based on which model the color belongs to
void GMM::addSample( int ci, const Vec3d color )
{
    //add color toghther for one model
    sums[ci][0] += color[0];
    sums[ci][1] += color[1];
    sums[ci][2] += color[2];
    
    
    //for cov
    prods[ci][0][0] += color[0]*color[0];
    prods[ci][0][1] += color[0]*color[1];
    prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0];
    prods[ci][1][1] += color[1]*color[1];
    prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0];
    prods[ci][2][1] += color[2]*color[1];
    prods[ci][2][2] += color[2]*color[2];
    
    //for modelWeight
    sampleCounts[ci]++;
    
    //each model, how many samples
    totalSampleCount++;
}

void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < modelNumInGMM; ci++ )
    {
        //calculate the modelWeight first based on sampleCounts
        int n = sampleCounts[ci];
        if( n == 0 )
            modelWeight[ci] = 0;
        
        else
        {
            modelWeight[ci] = (double)n/totalSampleCount;
            
            double* m = average + 3*ci;
            m[0] = sums[ci][0]/n;
            m[1] = sums[ci][1]/n;
            m[2] = sums[ci][2]/n;
            
            double* c = covariance + 9*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0];
            c[1] = prods[ci][0][1]/n - m[0]*m[1];
            c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0];
            c[4] = prods[ci][1][1]/n - m[1]*m[1];
            c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0];
            c[7] = prods[ci][2][1]/n - m[2]*m[1];
            c[8] = prods[ci][2][2]/n - m[2]*m[2];
            
            //double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
           
            
            calcInverseCovAndDet(ci);
        }
    }
}

void GMM::calcInverseCovAndDet( int ci )
{
    if( modelWeight[ci] > 0 )
    {
        double *c = covariance + 9*ci;
        double dtrm = covarianceDet[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
        
        //CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
        inverseCovariance[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovariance[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovariance[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovariance[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovariance[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovariance[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovariance[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovariance[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovariance[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}


// this function will return B
static double calculateB( const Mat& img )
{
    double B = 0;
    for( int x = 0; x < img.rows; x++ )
    {
        for( int y = 0; y < img.cols; y++ )
        {
            Vec3d color = img.at<Vec3b>(x,y);
            if( y>0 ) // left
            {
                Vec3d diffColor = color - (Vec3d)img.at<Vec3b>(x,y-1);
                B += diffColor.dot(diffColor);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diffColor = color - (Vec3d)img.at<Vec3b>(x-1,y-1);
                B += diffColor.dot(diffColor);
            }
            if( y>0 ) // up
            {
                Vec3d diffColor = color - (Vec3d)img.at<Vec3b>(x-1,y);
                B += diffColor.dot(diffColor);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diffColor = color - (Vec3d)img.at<Vec3b>(x-1,y+1);
                B += diffColor.dot(diffColor);
            }
        }
    }
    B = 1.0f / (2 * B/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );
    
    return B;
}

// second item in the function
//the function is for each pixel (item1+item2)
//store each node's (not including s and t) neighbors' information is 4 different matrix
static void calcWeight( const Mat& img, Mat& leftWeight, Mat& upleftWeight, Mat& upWeight, Mat& uprightWeight, double B, double gamma )
{
    double gammaWithSqrt= gamma / sqrt(2.0f);
    //use four mat to store the left, upleft, up and up right weight for each node(pixel) in the image
    leftWeight.create( img.rows, img.cols, CV_64FC1 );
    upleftWeight.create( img.rows, img.cols, CV_64FC1 );
    upWeight.create( img.rows, img.cols, CV_64FC1 );
    uprightWeight.create( img.rows, img.cols, CV_64FC1 );
    for( int x = 0; x < img.rows; x++ )
    {
        for( int y = 0; y < img.cols; y++ )
        {
            Vec3d color = img.at<Vec3b>(x,y);
            if( y-1>=0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(x,y-1);
                leftWeight.at<double>(x,y) = gamma * exp(-B*diff.dot(diff));
            }
            else
                leftWeight.at<double>(x,y) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(x-1,y-1);
                upleftWeight.at<double>(x,y) = gammaWithSqrt * exp(-B*diff.dot(diff));
            }
            else
                upleftWeight.at<double>(x,y) = 0;
            if( x-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(x-1,y);
                upWeight.at<double>(x,y) = gamma * exp(-B*diff.dot(diff));
            }
            else
                upWeight.at<double>(x,y) = 0;
            if( y+1<img.cols && x-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(x-1,y+1);
                uprightWeight.at<double>(x,y) = gammaWithSqrt * exp(-B*diff.dot(diff));
            }
            else
                uprightWeight.at<double>(x,y) = 0;
        }
    }
}

/*
 Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "type of mask is wrong" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "size of mask is wrong" );
    for( int x = 0; x < mask.rows; x++ )
    {
        for( int y = 0; y < mask.cols; y++ )
        {
            uchar type = mask.at<uchar>(x,y);
            if( type!=BG && type!=FG && type!=PBG && type!=PFG )
            {
                CV_Error( CV_StsBadArg, "mask elements are wrong");
            }
        }
    }
    
}


// Initialize mask using rectangular.
 
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( BG );
    mask(rect).setTo( PFG );
}


//Initialize GMM background and foreground models using kmeans algorithm.
 
static void initGMMs( const Mat& img, const Mat& mask, GMM& bgGMM, GMM& fgGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;
    
    Mat bgClustered, fgClustered;
    vector<Vec3f> bgSamples, fgSamples;
    
    //for each pixel in the image, put those BG in bgsample, FG in fgsample
    for( int x = 0; x < img.rows; x++ )
    {
        for( int y = 0; y < img.cols; y++ )
        {
            if( mask.at<uchar>(x,y) == BG || mask.at<uchar>(x,y) == PBG )
                bgSamples.push_back( (Vec3f)img.at<Vec3b>(x,y) );
            else // GC_FGD | GC_PR_FGD
                fgSamples.push_back( (Vec3f)img.at<Vec3b>(x,y) );
        }
    }
    Mat bgdSamplesMat( (int)bgSamples.size(), 3, CV_32FC1, &bgSamples[0][0] );
    kmeans( bgdSamplesMat, GMM::modelNumInGMM, bgClustered, TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat fgdSamplesMat( (int)fgSamples.size(), 3, CV_32FC1, &fgSamples[0][0] );
    kmeans( fgdSamplesMat, GMM::modelNumInGMM, fgClustered, TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    
    bgGMM.initLearning();
    for( int i = 0; i < (int)bgSamples.size(); i++ )
        bgGMM.addSample( bgClustered.at<int>(i,0), bgSamples[i] );
    bgGMM.endLearning();
    
    fgGMM.initLearning();
    for( int i = 0; i < (int)fgSamples.size(); i++ )
        fgGMM.addSample( fgClustered.at<int>(i,0), fgSamples[i] );
    fgGMM.endLearning();
}

/*
 Assign GMMs components for each pixel.
 */
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgGMM, const GMM& fgGMM, Mat& pixelModel )
{
    //Point p;
    for( int x = 0; x < img.rows; x++ )
    {
        for( int y = 0; y < img.cols; y++ )
        {
            Vec3d color = img.at<Vec3b>(x,y);
            if(mask.at<uchar>(x,y) == BG || mask.at<uchar>(x,y) == PBG )
            {
                pixelModel.at<int>(x,y) = bgGMM.whichModel(color);
            }
            else
            {
                pixelModel.at<int>(x,y) = fgGMM.whichModel(color);
            }
        }
    }
}

/*
 Learn GMMs parameters.
 */
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& pixelModel, GMM& bgGMM, GMM& fgGMM )
{
    bgGMM.initLearning();
    fgGMM.initLearning();
    //Point p;
    for( int ci = 0; ci < GMM::modelNumInGMM; ci++ )
    {
        for( int x = 0; x < img.rows; x++ )
        {
            for( int y = 0; y < img.cols; y++ )
            {
                if( pixelModel.at<int>(x,y) == ci )
                {
                    if( mask.at<uchar>(x,y) == BG || mask.at<uchar>(x,y) == PBG )
                        bgGMM.addSample( ci, img.at<Vec3b>(x,y) );
                    else
                        fgGMM.addSample( ci, img.at<Vec3b>(x,y) );
                }
            }
        }
    }
    bgGMM.endLearning();
    fgGMM.endLearning();
}

/*
 Construct GCGraph
 */
static void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgGMM, const GMM& fgGMM, double lambda,
                             const Mat& leftWeight, const Mat& upleftWeight, const Mat& upWeight, const Mat& uprightWeight,
                             GCGraph<double>& graph )
{
    //nodes number
    int nodeNum = img.cols*img.rows;
    //edge number
    int edgeNum = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
    
    
    // create the graph by using these nodes and edges
    graph.create(nodeNum, edgeNum);
    
    //For each point(x,y) in the graph;
    for( int x = 0; x < img.rows; x++ )
    {
        for( int y = 0; y < img.cols; y++)
        {
            // for each Node
            int nodeID = graph.addVtx();
            Vec3b color = img.at<Vec3b>(x,y);
            
            // set t-weights, the first item in the Gibbs formula
            double Souce, Sink;
            if( mask.at<uchar>(x,y) == PBG || mask.at<uchar>(x,y) == PFG )
            {
                Souce = -log( bgGMM(color) );
                Sink = -log( fgGMM(color) );
            }
            
            else if( mask.at<uchar>(x,y) == BG )
            {
                Souce = 0;
                Sink = lambda;
            }
            
            else if( mask.at<uchar>(x,y) == FG )
            {
                Souce = lambda;
                Sink = 0;
            }
            //Souce and Sink for Each Node
            graph.addTermWeights( nodeID, Souce, Sink );
            
            // set n-weights, the second item in the Gibbs formula
            if( y>0 )
            {
                double weight = leftWeight.at<double>(x,y);
                graph.addEdges( nodeID, nodeID-1, weight, weight );
            }
            if( x>0 && y>0 )
            {
                double weight = upleftWeight.at<double>(x,y);
                graph.addEdges( nodeID, nodeID-img.cols-1, weight, weight );
            }
            if( x>0 )
            {
                double weight = upWeight.at<double>(x,y);
                graph.addEdges( nodeID, nodeID-img.cols, weight, weight );
            }
            if( y<img.cols-1 && x>0 )
            {
                double weight = uprightWeight.at<double>(x,y);
                graph.addEdges( nodeID, nodeID-img.cols+1, weight, weight );
            }
        }
    }
}


// update the graph, return the new mask
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    //do max flow on the graph
    graph.maxFlow();
    
    //update the mask based on the maxflow result. update the PFG and PBG
    for( int x = 0; x < mask.rows; x++ )
    {
        for( int y = 0; y < mask.cols; y++ )
        {
            if( mask.at<uchar>(x,y) == PBG || mask.at<uchar>(x,y) == PFG )
            {
                if( graph.inSourceSegment( x*mask.cols+y ) )
                    mask.at<uchar>(x,y) = PFG;
                else
                    mask.at<uchar>(x,y) = PBG;
            }
        }
    }
}

//input image, maskimage which can tell us the FG and BG, rect(User choose), two GMM models and iterator number 

void grabCutMainFunction( Mat image, Mat& maskImage, Rect rect,
                 Mat& myBgModel, Mat& myFgModel,
                 int iterCount )
{
    //image mat
    Mat img = image;
    
    //check whether image is empty
    if( img.empty() )
        CV_Error( CV_StsBadArg, "image read fail" );
    
    //check the image type
    if( img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "the type of image is wrong" );
    
    //create 2 GMM model for background and forground, each GMM has 5 gas model and each model has a modelsize
    // two GMM model: myBgModel and myFgModel
    GMM bgGMM( myBgModel ), fgGMM( myFgModel );
    
    //model information for each pixel in the image
    Mat pixelModel( img.size(), CV_32SC1 );
    
    //imitial mask, set the rectangle to PFG
    initMaskWithRect( maskImage, img.size(), rect );
    
    //check whether this mask is correct
    checkMask( img, maskImage );
    
    
    //chushihua two GMM
    initGMMs( img, maskImage, bgGMM, fgGMM );

    //calculate three parameters in grabcut
    double gamma = 20;
    double lambda = 10*gamma;
    double B = calculateB(img);
    
    //edge weights which are stored in the  
    Mat leftWeight, upleftWeight, upWeight, uprightWeight;
    calcWeight( img, leftWeight, upleftWeight, upWeight, uprightWeight, B, gamma );
    
    //four step of grab cut
    for( int i = 0; i < iterCount; i++ )
    {
        GCGraph<double> graph;
        //assign GMM model for each pixel in the image, has result in the pixelModelMat
        assignGMMsComponents( img, maskImage, bgGMM, fgGMM, pixelModel );
        //set Model parameters from the sample
        learnGMMs( img, maskImage, pixelModel, bgGMM, fgGMM );
        constructGCGraph(img, maskImage, bgGMM, fgGMM, lambda, leftWeight, upleftWeight, upWeight, uprightWeight, graph );
        //update the mask based the max flow on graph, get new maskImage
        estimateSegmentation( graph, maskImage );
    }
}


//line's color
const Scalar BLUE = Scalar(255,0,0);


static void getNewMask( Mat& maskImage, Mat& newMask )
{
    if( maskImage.empty() || maskImage.type()!=CV_8UC1 )
        CV_Error( CV_StsBadArg, "Mask is wrong!" );
    if( newMask.empty() || newMask.rows!=maskImage.rows || newMask.cols!=maskImage.cols )
        newMask.create( newMask.size(), CV_8UC1 );
    newMask = maskImage & 1;
}




//GrabCut GUI Class

class GrabCut
{
public:
    //three state
    enum
    {
        NO = 0, PROCESS = 1, SET = 2
    };
    //reset function
    void reset();
    //initial window and load the image
    void initialWindow( const Mat& myImage, const string& myWinName );
    //show the image
    void showImage();
    //mouse acrion
    void mouseGrab( int event, int x, int y, int flags, void* param );
    //interate
    int Iterator();
    //get the iterate number
    int getIterCount();
    int rectState;
    Rect rect;
private:
    void setRectInMask();
    const string* winName;
    const Mat* image;
    Mat maskImage;
    Mat bgdModel, fgdModel;
    
    
    
    //this valurable is for determing whether user has pressed the key, the default valur is false, user has not pressed the key
    bool pressKey;
    int iterCount;
};

void GrabCut::reset()
{
    if(!maskImage.empty())
    {
        maskImage.setTo(Scalar(BG));
    }
    pressKey = false;
    rectState = NO;
    iterCount = 0;
}


//create image, window and mask of the image. Mask can determin the class of the pixel
void GrabCut::initialWindow( const Mat& myImage, const string& myWinName  )
{
    if(myImage.empty() || myWinName.empty())
    {
        return;
    }
    image = &myImage;
    winName = &myWinName;
    maskImage.create( image->size(), CV_8UC1); //create the mask. The size is the same as the image
    reset();
}

void GrabCut::showImage() 
{
    if(image->empty() || winName->empty())
    {
         return;
    }
    Mat orignalImage;
    Mat newMask;
    if( !pressKey )
        image->copyTo(orignalImage);
    //key for begining iteration has been pressed
    else
    {
        getNewMask( maskImage, newMask );
        image->copyTo( orignalImage, newMask );
    }
    if( rectState == PROCESS || rectState == SET )
        rectangle( orignalImage, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), BLUE, 10);
    
    imshow( *winName, orignalImage );
}

void GrabCut::setRectInMask()
{
    assert( !maskImage.empty() );  //make sure that mask is not empty. if it is empty, we have set it to background 
    maskImage(rect).setTo( PFG); //set the rectangle part to forward
}


void GrabCut::mouseGrab( int event, int x, int y, int flags, void* )
{
    switch( event )
    {
        case CV_EVENT_LBUTTONDOWN:
        {
            if( rectState == NO  )
            {
                rectState = PROCESS;
                rect = Rect( x, y, 0, 0 );
            }
             break;
        }
           
        case CV_EVENT_LBUTTONUP:
            if( rectState == PROCESS )
            {
                //rect.width and rect.height stroe the information of the rectangle
                rect = Rect( Point(rect.x, rect.y), Point(x,y) );
                cout<<"The width of image is "<<rect.width<<endl;
                cout<<"The Height of image is "<<rect.height<<endl;
                rectState = SET;
                setRectInMask();
                showImage();
            }
            break;
        case CV_EVENT_MOUSEMOVE:
            if( rectState == PROCESS )
            {
                rect = Rect( Point(rect.x, rect.y), Point(x,y) );
                showImage();
            }
            break;
    }
}

int GrabCut::Iterator()
{
    //call grab cut algorithm
    grabCutMainFunction( *image, maskImage, rect, bgdModel, fgdModel, 1);
    if( !pressKey )
    {
        pressKey = true;
    }
    iterCount++;
    return iterCount;
}
int GrabCut::getIterCount()
{
    return iterCount;
}


GrabCut mainObject;

void mouseOperation( int event, int x, int y, int flags, void* param )
{
    mainObject.mouseGrab( event, x, y, flags, param );
}

int main()
{
    string file = "/Users/yangbo/Downloads/OpenCVTest-3/lena.jpg";
    if( file.empty() )
    {
        cout << "file read fail! " << file << endl;
        return 1;
    }
    Mat image = imread( file, 1 );
    if( image.empty() )
    {
        cout << "image read fail" << file << endl;
        return 1;
    }
    string window = "GrabCut";
    
    //define the window's name
    cvNamedWindow( window.c_str(), CV_WINDOW_AUTOSIZE );
    //mouse action
    cvSetMouseCallback( window.c_str(), mouseOperation, 0 );
    //create image and mask
    mainObject.initialWindow( image, window );
    mainObject.showImage();
    
    while(1)
    {
        int key = cvWaitKey(0);
        char operation = (char)key;
        switch(operation)
        {
            //end the program
            case '\x1b':
                cvDestroyWindow( window.c_str() );
                return 0;
            //end this iteration
            case 'e':
                mainObject.reset();
                mainObject.showImage();
                break;
            //begin a iteration
            case 'b':
                int iterCount = mainObject.getIterCount();
                cout << "Previous Iteration is " << iterCount<<endl;
                if(mainObject.rectState != mainObject.SET||mainObject.rect.width==0||mainObject.rect.height==0)
                {
                     cout << "please choose you region!" << endl;
                }
                else
                {
                    cout << "Iteration " << iterCount+1<<" Processing............."<<endl;
                    //run grab cut aldorithm
                    int newIterCount = mainObject.Iterator();
                    cout << "Iteration " << newIterCount<<" Finished"<<endl;
                    cout << "Current Iteration is " << newIterCount<<endl;
                    mainObject.showImage();
                }
                break;
        }
    }
    return 0;
}

