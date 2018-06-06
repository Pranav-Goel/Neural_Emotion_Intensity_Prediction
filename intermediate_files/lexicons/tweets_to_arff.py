# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# tweets_to_arff.py
# felipebravom
# Running example: python tweets_to_arff data/anger-ratings-0to1.test.target.tsv data/anger-ratings-0to1.test.target.arff
import sys


def create_arff(input_file,output_file):
    """
    Creates an arff dataset
    """
    
    


    out=open(output_file,"w")  
    header='@relation '+input_file+'\n\n@attribute id numeric \n@attribute tweet string\n@attribute emotion string\n@attribute score numeric \n\n@data\n'
    out.write(header)
       

       
    f=open(input_file, "rb")
    lines=f.readlines()
    
    
    for line in lines:
        parts=line.split("\t")
        if len(parts)==4:
     
            id=parts[0]
            tweet=parts[1]
            emotion=parts[2]
            score=parts[3].strip() 
            score = score if score != "NONE" else "?"
            
            out_line=id+',\"'+tweet+'\",'+'\"'+emotion+'\",'+score+'\n'
            out.write(out_line)
        else:
            print "Wrong format"
    

    f.close()  
    out.close()  
    
    
    
  
    
def main(argv):
    input_file=argv[0]
    output_file=argv[1]
    create_arff(input_file,output_file)
   
        
if __name__ == "__main__":
    main(sys.argv[1:])    
    




    

    
    