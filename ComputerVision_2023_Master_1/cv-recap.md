# Chapter 1
* What is vision
	* it is more complex that pixel-input understanding
	* context, intensity, brightness
* Application examples for CV
	* medical image processing
	* self-driving cars
	* face generation
	* augmented really
	* surveillence cameras
	* motion capture (for games)
* Light
	* essential for CV
	* it's a spectre composed of wave lengths, phases, polarizing directions
	* visible ranges (in nm):
        * 380 - 450 violet
        * 450 - 490 blue
        * 490 - 560 green
        * 560 - 590 yellow
        * 590 - 630 orange
        * 630 - 760 red
    * Interactions with matter
        * absorption : blue water
        * scattering (3 types depending on relative sizes of particles and wavelengths) : blue sky, red sunset
            * small particles: Rayleigh (strongly wavelength dependent)
            * comparable sizes: Mie (weakly wavelength dependent)
            * Large particles: non-selective (wavelength independent)
        * reflection : coloured ink
            * Angle of reflection = angle of incidence
            * Roughness of surfaces leads to `diffuse’ reflection
                * most surfaces are rough
        * refraction : dispersion by a prism
            * Snells law: n1 * sinΔ_i = n2 * sinΔ_t
        * Refraction is more complicated than mirror reflection: the path orientation of light rays is changed depending on material AND wavelength

# Chapter 2 (Acquisition of images)
* focused on illumination and cameras
* illumination:
    * a well designed illumination is key to visual inspection
    * illumination techniques: 
        * back-lighting
            * lamps placed behind a transmitting diffuser plate, light source behind the object
            * generates high-contrast silhouette images, easy to handle with binary vision
            * often used in inspection
        * directional-lighting
            * generate sharp shadows
            * generation of specular reflection (for example used in crack detection)
            * shadows and shading yield information about shape
        * diffuse-lighting
            * illuminates uniformly from all directions
            * prevents sharp shadows and large intensity variations over glossy surfaces
            * all directions contribute extra diffuse reflection, but contributions to the specular peak arise from directions close to the mirror one only
        * polarized-lighting
            * to improve contrast between Lambertian and specular reflections
            * to improve contrasts between dielectrics and metals, e.g. when inspecting electrical circuits
            * Polarization direction is the one of the E-wave.
            * Normally, the light is composed of many waves with different polarizations
        * coloured-lighting
            * highlight regions of a similar colour
            * with band-pass filter: only light from projected pattern (e.g. monochromatic light from a laser)
            * differentiation between specular and diffuse reflection
            * comparing colours
            * spectral sensitivity function of the sensors
        * structured-lighting
            * 3D shape : objects distort the projected pattern
        * stroboscopic lighting
            * high intensity light flash
            * to eliminate motion blur
* cameras
    * the pinhole model (image upside down)
    * linear magnification formula
    * the thin-lens equation
    * The depth-of-field
        * microscopes -> small DoF
    * abberations
        * geometrical: visible as image distortions or degradation like blurring
            * spherical aberration
                * rays parallel to the axis do not converge
                * outer portions of the lens yield smaller focal lengths
            * radial distortion (different magnification for different angles of inclination)
                * The result is lines become curves.
                * curveture increases as you move away from the center of distortion.
            * astigmatism
            * coma
        * chromatic: visible as different behavior for different wavelengths (e.g. colors)
            * rays of different wavelengths focused in different planes
            * cannot be removed completely, but achromatization can be achieved at some well chosen wavelength pair, by combining lenses made of different glasses
    * types of cameras
        * CCD (Charge-coupled device)
            * high end
        * CMOS (Complementary Metal Oxide Semiconductor)
            * Same sensor elements as CCD
            * Each photo sensor has its own amplifier
            * More noise
            * Lower sensitivity
            * Allows to put other components on chip
            * 'Smart' pixels
            * common in mobile phones
    * types of color cameras
        * Prism (with 3 sensors)
        * Filter mosaic
        * Filter wheel
            * Only suitable for static scenes
    * Geometric camera model

# Chapter 3
* color
    * The perception of brightness
    * The study of colour
        * pleasing to the eye (visualisation of results)
        * characterising colours (features e.g. for recognition)
        * generating colours (displays, light for inspection)
        * understanding human vision
    * BHS = Brightness, hue, saturation
    * history of color
        * spectrum (Newton)
        * tristimulus model (Young)
    * CIE chromaticity diagram
        * r = R / (R + G + B)
    * Colour constancy
        * Patches keep their colour appearance even if they reflect differently (e.g. the hat)
        * Patches change their colour appearance if they reflect identically but surrounding patches reflect differently (e.g. the background)
        * There is more to colour perception than 3 cone responses
        * The colour of a surface is the result of the product of spectral reflectance and spectral light source composition
    * Bidirectional Reflection Distribution Function
        * A 4D function, specifying the radiance for an outgoing direction given an irradiance for an incoming direction, relative to the normal and ideally for 1 wavelength at a time
* THE FOURIER TRANSFORM COLLECTS INFORMATION GLOBALLY OVER THE ENTIRE IMAGE. NOT GOOD FOR SEGMENTATION OR INSPECTION
* Histograms
    * Intensity probability distribution
    * Captures global brightness information in a compact, but incomplete way
    * more like a frequency vector (vector de frecventa)
* Cooccurrence matrix
    * probability distributions for intensity pairs
    * Contains information on some aspects of the spatial configurations
* Laws filters
* Gabor filters
* Eigenfilters
    * shift mask over training image
    * collect intensity statistics
    * Filters adapted to the texture, but small filters may reduce efficacy, hence large, but sparse filters
    * Example applications: textile inspection


# Chapter 4
* Sampling and quantization
    * sampling leads to pixels
    * quantisation leads to "grey levels" (how many "types" of gray are in a given image)
    * Fourier Transform has peaks at spatial frequencies of repeated texture
    * The convolution theorem states that: space convolution = frequency multiplication
        * the reciprocity also holds: space multiplication = frequency convolution
    * The sampling theorem
        * If the Fourier transform of a function f(x,y) is zero for all frequencies beyond ub and vb, i.e. if the Fourier transform is band-limited, then the continuous periodic function f(x,y) can be completely reconstructed from its samples as long as the sampling distances w and h along the x and y directions are such that $w <= 1/2u_b$ and $h <= 1/2v_b$
    * Quantization
        * Create K intervals in the range of possible intensities measured in bits: $log_2(K)$
        * you can create different types of intervals (for example, the simples one is equal intervals)
        * Often 8 bits per pixel (monochrome), 24 bits per pixel (RGB)
        * Medical images 12 bits (4096 levels) or 16 bits (65536 levels)
* image enhancement
    * Noise suppression
        * specific methods for specific types of noise
        * only consider 2 general options
            * Convolutional linear filters (low-pass convolution filters)
                * Goal: remove low-signal/noise part of the spectrum
                * Approach 1: Multiply the Fourier domain by a mask
                * Approach 2: generate low-pass filters that do not cause rippling
                * the simplest filter example is the average filter
                * other examples:
                    * binomial filter
                    * Gaussian filter
                * Actually linear filters cannot solve this problem
            * Non-linear filters (edge-preserving filters)
                * Median
                    * there is a two step method:
                        1. rank-order neighbourhood intensities
                        2. take middle value
                    * this results in no more grey levels
                    * advantage of this type of filter is its “odd-man-out” effect (1,1,1,7,1,1,1,1 -> ?,1,1,1.1,1,1,?)
                    * median completely discards the spike, linear filter always responds to all aspects
                    * median filter preserves discontinuities, linear filter produces rounding-off effects
                * Anisotropic diffusion
                    * there is a two step method:
                        1. Gaussian smoothing across homogeneous intensity areas
                        2. No smoothing across edges
                    * End state is homogeneous
                    * Can also have a numerical solution
    * Image de-blurring
        * Unsharp masking
            * simple, effective
            * image independent
            * linear
        * Inverse filtering
            * Relies on system view of image processing
            * Frequency domain technique
            * Defined through Modulation Transfer Function
            * Links to theoretically optimal approaches
        * The Wiener Filter
            * Looking for the optimal filter to do the deblurring
            * Take into account the noise to avoid amplification
            * Optimization formulation
            * Filter is given analytically in the Fourier Domain
    * Contrast enhancement
        * multiple uses:
            * compensating under-, overexposure
            * spending intensity range on interesting part of the image
        * Intensity distribution
        * Intensity mappings
        * Gamma correction
        * HISTOGRAM EQUALISATION
            * Redistribute the intensities, 1-to-several (1-to-1 in the continuous case) and keeping their relative order, as to use them more evenly
            * Ideally, obtain a constant, flat histogram

# Chapter 5
* Feature matching
    * Feature: measured characteristic of (part of) a pattern / object
    * Goal : efficient matching for
    * difficulties: Deformation, illumination change, occlusion, Large scale change, perspective deformation, extensive clutter, scale, occlusion, blur
    * for short, matching is a challenging task because of large variations in: viewpoints, illumination, background and occlusion (obstacles that block the view to the main object)
    * feature consideration:
        * Complete ( describing pattern unambiguously) or not
        * Robustness of extraction
        * Ease of extraction
        * Global vs. local
    * A feature should capture something discriminative about a well localisable patch of a surface
* The Harris corner detector
* Corners are the most prominent example of so-called **Interest Points**, i.e. points that can be well localised in different views of a scene.
* **Blob** is a region with intensity changes in multiple directions
* Need for an invariant descriptor:
    * There are many corners coming out of our DETECTOR, but they still cannot be told apart
    * We need to describe their surrounding image patch such we can discriminate between them, i.e. we need to build a feature vector for the patch, a so-called DESCRIPTOR
    * During a MATCHING step, the descriptors can then be compared. In order for that to be easy, the descriptors for corresponding, detected points must be similar in different views. i.e. invariant under the changes between these views.
    * Deformations under projection (how do different views of the same planar shape / contour differ):
        * viewed from a perpendicular direction
        * viewed from any direction but at sufficient distance to use pseudo - perspective
        * viewed from any direction and from any distance
    * Invariance under transformations implies invariance under the smallest encompassing group
    * photometric changes
        * Contrasts (intensity differences) let the non-linear offsets cancel; hence gradients are good !
        * Moreover the orientation of gradients in the color bands is invariant under their linear changes, as is the intensity gradient orientation in case the scale factors are identical; this is indeed relevant if the illumination changes its intensity, but not its color, which is typically assumed.
        * But even under changing color of the illumination, in practice edge orientations tend to remain the same.
    * the final goal: define good interest points (DETECTORS + DESCRIPTORS)
        * The detector typically yields image points
        * Descriptors then are a vector of measurements taken around each such point
* Interest points are matched on the basis of their descriptors
* The shape of the patch should change with viewpoint
* The important thing is to achieve such change in patch shape without having to compare the images, i.e. this should happen on the basis of information in one
* MSER interest points (Maximally Stable Extremal Regions)
    * Similar to the Intensity-Based Regions
    * Came later, but is more often used
    * Start with intensity extremum
    * Then move intensity threshold away from its value and watch the super/sub-threshold region grow
    * Take regions at thresholds where the growth is slowest (happens when region is bounded by strong edges)
* SIFT = Scale-Invariant Feature Transform
    * is a carefully crafted interest point detector + descriptor, based on intensity gradients and invariants under similarities, not affine like so far
* SURF efficient alternative to SIFT

# Chapter 6 (Segmentation (Identifying entities in the image))
* Can be viewed as grouping pixels into segments
* Thresholding
    * For high contrast between object(s) and background, determine intensity threshold that defines 2 pixel categories : object and background
    * Alternatives:
        * If intensities of objects are known, then it's easy
        * From histogram, take the minimum between two peaks
        * For known size (e.g. for industrial application), increase threshold until reaching a predefined area
        * Maximize sum of gradients at pixels with threshold intensity
        * Low gradient magnitude areas
        * Use regions with high response to Laplacian filter – points around the edge
    * Otsu criterion
        * For short, can be used to set a varying threshold
    * Binary enhancement
        * Erosion + dilation (opening), Dilation + erosion (closing)
        * Use the same structural element for both steps
        * It is a post-processing approach (It has many alternatives that enforce neighborhood consistency during segmentation)
        * median filtering very useful and commonly used as edge preserving smoothing
    * Advantages:
        * Serious bandwidth reduction
        * Simplification for further processing
        * Availability of real-time hardware
    * Generally it won’t provide a satisfying segmentation
    * Pixel-by-pixel decision ignores neighbouring pixels and structural information lost
* Edge based
    * Edges are useful to infer shape and occlusion
    * Edge linking techniques:
        * Hough Transform: for predefined shapes
            * Is a voting technique that can tell how many lines are there, which points belong to what lines, where are the lines given the points
            * idea: Record vote for each possible line on which each edge point lies. Look for lines that get many votes.
            * a point in the image -> a line in the parameter space
            * a line in the image -> a point in the parameter space
            * practical tips:
                * Minimize irrelevant voting points
                * Choose a good discretization / grid
                * Vote for neighbors, also (smoothing in accumulator)
                * Weight the votes (e.g. by intensity gradient magnitude)
                * Use everything you know (e.g. direction of the edge)
            * pros:
                * All points are processed independently, so can cope with occlusions, gaps
                * Some robustness to noise: noise points unlikely to affect the voting outcome
                * Can detect multiple instances of a shape model in a single pass
            * cons:
                * Time complexity exponential in number of model parameters
                * Ambiguities are possible (spurious peaks) - if similar shapes are close by
                * Discretization: can be tricky to pick a good grid size
                * “Good” peak detection is nontrivial
        * Elastically deformable contour models Snakes: generic shape priors
            * Given: initial contour (model) near desired object
            * Goal: evolve the contour to fit the object boundary
            * Intuition: an elastic (rubber) band wrapping around structures to cover / fill-in missing parts
            * only “see” nearby object boundaries
            * Define a curve as a set of n points, an internal deformation and an external image-based energy
            * Initialize “near” object boundary, and iteratively optimize the curve points to minimize the total energy
            * pros:
                * Useful to fit non-rigid arbitrary prior shapes in images
                * Contour remains connected, i.e. topology is fixed
                * Possible to connect / fill in invisible contours
                * Flexibility in energy function definition, i.e., allows other forces and interactive input
            * cons:
                * Local optimization: may get stuck in local minimum. Thus, needs good initialization near true boundary
                * Susceptible to parameterization of energy function, must be set based on prior information, experience, etc.
        * Many other methods for grouping combination with user interaction
* Region based
    * On the basis of segment homogeneity rather than inhomogeneity around edges
    * start with detection of “homogeneous” regions (e.g. low intensity variance) as the “seeds”
    * These are grown as long as homogeneity criterion is satisfied
    * Choice of appropriate homogeneity criteria is not straightforward
    * Region and edge based methods can be combined: hybrid approaches
    * Watershed algorithm
* Statistical Pattern Recognition based
    * General scheme
    * Feature based
    * Probabilistic and learning based formulations
    * alternatives: 
        * Unsupervised clustering 
            * K-means
        * Supervised generative modeling
            * Assumes existing examples where we can learn distributions for measurements and classes
            * From individual pixels to combinations
        * Supervised discriminative modeling
            * KNN
            * Random Forests

# Chapter 7 (Traditional object recognition)
* problem - classification - detection
* Specific object recognition: Identifying the same object in different images despite variations in pose and illumination
    * A specific object = an instance of an object class ("my car", not "a car")
    * app example: mobile tour guide for landmarks
    * Challenges
        * Pose / viewpoint changes
        * Illumination variation
        * Occlusion
        * Clutter
    * Model-based approaches
    * Invariant-based recognition of planar shapes
        * The crucial advantage of invariants is that they decouple object type and pose issues
    * Appearance based methods (The model of an object is simply its image(s))
    * The problem is variation in the appearance because of changes in viewpoint / lighting
    * Object-pose manifold
    * Comparison between model-based and appearance-based techniques
        * Pure model-based
            * Compact model
            * Can deal with clutter
            * Slow analysis-by-synthesis
            * Models difficult to produce
            * Models difficult to produce
        * Pure appearance-based
            * Large models
            * Cannot deal with clutter
            * Efficient
            * Models easy to produce
            * For wide classes of objects
    * Image-based – global image representation
    * Hybrid models – local image representations
        * Detection: Identify interest points
        * Description: Extract feature vector descriptors around them
        * Matching: Determine correspondence between descriptors in two views
* Object category recognition: Identifying the object category despite variations in pose, illumination and intra-class variation
    * classifcation: is there a car in this image ? A binary answer is enough
    * detection: where is the car ? Need localization
    * visual word
    * A bag of words is an orderless representation: discarding spatial relationships between features
        * pros:
            * Flexible to geometry / deformations / viewpoint
            * Compact summary of image content
            * Provides vector representation for sets
            * Empirically good recognition results in practice
        * cons
            * Basic model ignores geometry – can verify afterwards, or embed within feature descriptors
            * Background and foreground mixed when bag covers whole image
            * Interest points or sampling: no guarantee to capture object-level parts
            * Optimal vocabulary formation remains unclear
    * Spatial Pyramid description matching
    * sliding windows
            * Rectangular Integral Image Filters
            * AdaBoost (example: Viola-Jones Face Detector)
            * pros:
            * Simple detection protocol to implement
            * Good feature choices critical, but part of the process
            * Past successes for certain classes
            * Good detectors available (Viola&Jones, HOG, etc.)
        * cons:
            * High computational complexity
            * With so many windows, false positive rate better be low
            * Typically need fully supervised training data (= bounding-boxes)
            * Objects with less-regular textures not captured well
            * Non-rigid, deformable objects not captured well
    * Generalized Hough Transform (Implicit Shape Model)
        * pros:
            * Works well for many different object categories
            * Flexible geometric model
            * Learning from relatively few (50-100) training examples
            * Optimized for detection, good localization properties
        * cons
            * Needs supervised training data
            * Only weak geometric constraints
            * Purely representative model

# Chapter 8 (Tracking)
* application: surveillance, sports, driving, video editing, VR, medical guidance, Gesture/Action Recognition
* what to track: center point / multiple points / structure / body parts / region / outline
* how it (usually) works (simple tracking):
    * evaluate state at time T
    * predict T+1
    * evaluate state at time T+1
    * update the model
* Tracking-by-Detection evaluates each state independently
    * 3D Object Detection
        * Detect Keypoints (invariant to scale, rotation, or perspective)
        * Build Feature Descriptors
        * Match Keypoint Descriptors
    * associate detections over time into tracks
* Background Modeling : For known (fixed) background, simply save it and subtract from each frame
* Region tracking
* Point tracking
    * good image features (with large structural eigenvalues) are also good for tracking with which we can find motion
* Template tracking
    * Keep a template image to compare with each frame
    * Lucas-Kanade Template Tracker
* Model based tracking
* Only thing we are sure about the object is its initial model (first frame apperance). We can “anchor” / correct our model with this information, in order to help avoid drift

# Chapters 9 and 10 (Deep Learning intro)
* supervised vs unsupervised learning
* input data - labels
* cost function
* prediction
* perceptron / MLP
* train / validate / test sets

# Chapters 11 and 12 (Convolutional Neural Networks)
* efectiv CNN uri si tot ce inseamna ele: pooling, layers, kernels, stride, etc